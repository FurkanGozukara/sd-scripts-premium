"""
SDXL-specific CPU offloading utilities for block swap.
Adapted from OneTrainer's approach to handle SDXL's non-uniform UNet structure.
"""

import torch
import torch.nn as nn
from typing import List, Union
from library.custom_offloading_utils import ModelOffloader


def collect_transformer_blocks_from_unet(unet: nn.Module) -> List[nn.Module]:
    """
    Collect all BasicTransformerBlock instances from SDXL UNet.
    
    Works with:
    - Diffusers' BasicTransformerBlock  
    - Kohya's BasicTransformerBlock (in sdxl_original_unet.py)
    
    SDXL UNet structure:
    - input_blocks/down_blocks: List of blocks with attention
    - middle_block/mid_block: Middle block with attention
    - output_blocks/up_blocks: List of blocks with attention
    """
    transformer_blocks = []
    
    # Try to import BasicTransformerBlock from Kohya's sdxl_original_unet
    try:
        from library.sdxl_original_unet import BasicTransformerBlock as KohyaBasicTransformerBlock
        has_kohya_blocks = True
    except ImportError:
        has_kohya_blocks = False
        KohyaBasicTransformerBlock = None
    
    # Try to import BasicTransformerBlock from diffusers
    try:
        from diffusers.models.attention import BasicTransformerBlock as DiffusersBasicTransformerBlock
        has_diffusers_blocks = True
    except ImportError:
        has_diffusers_blocks = False
        DiffusersBasicTransformerBlock = None
    
    # Collect from all modules in the UNet
    for module in unet.modules():
        # Check for Kohya's BasicTransformerBlock first (most common in Kohya training)
        if has_kohya_blocks and isinstance(module, KohyaBasicTransformerBlock):
            transformer_blocks.append(module)
        # Check for diffusers BasicTransformerBlock
        elif has_diffusers_blocks and isinstance(module, DiffusersBasicTransformerBlock):
            transformer_blocks.append(module)
        # Fallback: check by class name for compatibility
        elif module.__class__.__name__ == 'BasicTransformerBlock':
            transformer_blocks.append(module)
    
    return transformer_blocks


def calculate_block_memory_sizes(blocks: List[nn.Module]) -> List[int]:
    """
    Calculate the memory size (in bytes) for each block.
    This is important for SDXL because blocks are NOT uniform in size.
    """
    block_sizes = []
    for block in blocks:
        total_bytes = 0
        for param in block.parameters():
            total_bytes += param.numel() * param.element_size()
        block_sizes.append(total_bytes)
    
    return block_sizes


def place_unet_for_block_swap(
    unet: nn.Module,
    transformer_blocks: List[nn.Module],
    blocks_to_swap: int,
    device: torch.device,
    debug: bool = False,
):
    """
    Move UNet parameters to the correct devices before wrapping with Accelerator.

    - Keeps the last `blocks_to_swap` transformer blocks on CPU
    - Moves the rest of the model (including non-transformer parts) to `device`
    - Avoids loading all blocks onto the GPU at once (prevents initial VRAM spike)
    """
    if blocks_to_swap <= 0 or len(transformer_blocks) == 0:
        unet.to(device)
        return

    num_blocks = len(transformer_blocks)
    keep_until = max(0, num_blocks - blocks_to_swap)
    blocks_to_keep = transformer_blocks[:keep_until]
    blocks_to_cpu = transformer_blocks[keep_until:]

    # Track parameters/buffers belonging to CPU blocks so we can skip moving them
    cpu_param_ids = set()
    cpu_buffer_ids = set()
    for block in blocks_to_cpu:
        for p in block.parameters(recurse=True):
            cpu_param_ids.add(id(p))
        for b in block.buffers(recurse=True):
            cpu_buffer_ids.add(id(b))

    # Move blocks explicitly
    for block in blocks_to_keep:
        block.to(device)
    for block in blocks_to_cpu:
        block.to(torch.device("cpu"))

    # Move remaining parameters/buffers that are not part of CPU blocks
    with torch.no_grad():
        for p in unet.parameters():
            if id(p) in cpu_param_ids:
                continue
            if p.device != device:
                p.data = p.data.to(device=device)

        for b in unet.buffers():
            if id(b) in cpu_buffer_ids:
                continue
            if isinstance(b, torch.Tensor) and b.device != device:
                b.data = b.data.to(device=device)

    if debug:
        print(
            f"[place_unet_for_block_swap] "
            f"kept {len(blocks_to_keep)} blocks on {device}, "
            f"left {len(blocks_to_cpu)} blocks on CPU"
        )


def estimate_optimal_blocks_to_swap(
    blocks: List[nn.Module],
    target_swap_fraction: float = 0.3,
    min_blocks_on_device: int = 2
) -> int:
    """
    Estimate optimal number of blocks to swap based on memory usage.
    
    Args:
        blocks: List of transformer blocks
        target_swap_fraction: Fraction of blocks to swap (0.0 to 1.0)
        min_blocks_on_device: Minimum number of blocks to keep on device
    
    Returns:
        Number of blocks to swap
    """
    total_blocks = len(blocks)
    blocks_to_swap = int(total_blocks * target_swap_fraction)
    
    # Ensure we keep minimum blocks on device
    max_swappable = total_blocks - min_blocks_on_device
    blocks_to_swap = min(blocks_to_swap, max_swappable)
    blocks_to_swap = max(0, blocks_to_swap)
    
    return blocks_to_swap


class SDXLBlockSwapManager:
    """
    Manages block swapping for SDXL UNet using OneTrainer-style approach.
    
    Unlike FLUX which has uniform double/single blocks, SDXL has:
    - Variable-sized transformer blocks
    - Blocks distributed across input_blocks, middle_block, and output_blocks
    - Different memory footprints per block
    
    This implementation adds forward hooks to move blocks to GPU before use,
    and backward hooks to move them back to CPU after gradients are computed.
    """
    
    def __init__(
        self,
        unet: nn.Module,
        blocks_to_swap: int,
        device: torch.device,
        debug: bool = False
    ):
        self.unet = unet
        self.device = device
        self.debug = debug
        
        # Collect all transformer blocks
        self.transformer_blocks = collect_transformer_blocks_from_unet(unet)
        self.num_blocks = len(self.transformer_blocks)
        
        if self.num_blocks == 0:
            raise ValueError("No transformer blocks found in UNet. SDXL block swap requires transformer blocks.")
        
        # Calculate memory sizes for adaptive swapping
        self.block_sizes = calculate_block_memory_sizes(self.transformer_blocks)
        total_memory = sum(self.block_sizes)
        
        # Validate blocks_to_swap
        max_swappable = self.num_blocks - 2  # Keep at least 2 blocks on device
        if blocks_to_swap > max_swappable:
            print(f"Warning: Requested {blocks_to_swap} blocks to swap, but only {max_swappable} can be safely swapped.")
            print(f"Reducing to {max_swappable} blocks.")
            blocks_to_swap = max_swappable
        
        self.blocks_to_swap = blocks_to_swap
        self.forward_handles = []
        self.backward_handles = []
        self.original_methods = {}
        
        # Register hooks for swapping
        if self.blocks_to_swap > 0:
            self._register_swap_hooks()
            
            if debug or True:  # Always print for now
                print(f"SDXL Block Swap initialized:")
                print(f"  Total transformer blocks: {self.num_blocks}")
                print(f"  Blocks to swap: {self.blocks_to_swap}")
                print(f"  Blocks on device: {self.num_blocks - self.blocks_to_swap}")
                print(f"  Total memory: {total_memory / (1024**3):.2f} GB")
                if len(self.block_sizes) >= self.blocks_to_swap:
                    print(f"  Memory on device: {sum(self.block_sizes[:-self.blocks_to_swap]) / (1024**3):.2f} GB")
                    print(f"  Memory to swap: {sum(self.block_sizes[-self.blocks_to_swap:]) / (1024**3):.2f} GB")
        else:
            print("SDXL Block Swap: No blocks to swap (blocks_to_swap = 0)")
    
    def _register_swap_hooks(self):
        """Register forward and backward hooks for automatic block swapping."""
        num_blocks_on_device = self.num_blocks - self.blocks_to_swap
        
        # Register hooks only on blocks that will be swapped (the last N blocks)
        for block_idx in range(num_blocks_on_device, self.num_blocks):
            block = self.transformer_blocks[block_idx]
            
            # Monkey-patch forward/forward_body to handle block swapping
            # This is necessary because register_forward_pre_hook is NOT called during the re-forward pass 
            # of gradient checkpointing, but the method itself IS called.
            
            if hasattr(block, 'forward_body'):
                # Kohya/Original SDXL UNet
                original_method = block.forward_body
                method_name = 'forward_body'
            else:
                # Diffusers/Other (fallback)
                original_method = block.forward
                method_name = 'forward'
                
            def create_swap_wrapper(original, b_idx, b_module, b_name):
                def wrapped_forward(*args, **kwargs):
                    if self.debug:
                        print(f"Swap Wrapper ({b_name}): block {b_idx} to {self.device}")
                    
                    # Swap In
                    b_module.to(self.device)
                    
                    try:
                        result = original(*args, **kwargs)
                    finally:
                        # Swap Out logic
                        # 1. If no_grad (real forward in checkpointing): Swap Out
                        # 2. If grad (re-forward): Keep on GPU for backward
                        if not torch.is_grad_enabled():
                            if self.debug:
                                print(f"Swap Wrapper ({b_name}): block {b_idx} offload (no_grad)")
                            b_module.to(torch.device("cpu"))
                    
                    return result
                return wrapped_forward
            
            # Save original method to restore later
            self.original_methods[block] = (method_name, original_method)
            
            # Apply patch
            setattr(block, method_name, create_swap_wrapper(original_method, block_idx, block, method_name))
            
            # Backward hook is still needed to cleanup after backward pass
            def backward_hook(module, grad_input, grad_output, block_idx=block_idx):
                if self.debug:
                    print(f"Backward hook: Moving block {block_idx} back to CPU")
                if self.device.type == "cuda":
                    torch.cuda.synchronize()
                module.to(torch.device("cpu"))
                return None
            
            bwd_handle = block.register_full_backward_hook(backward_hook)
            self.backward_handles.append(bwd_handle)
    
    def prepare_block_swap_before_forward(self):
        """Prepare block devices before forward pass."""
        if self.blocks_to_swap > 0:
            num_blocks_on_device = self.num_blocks - self.blocks_to_swap
            
            if self.debug:
                print(f"Preparing {num_blocks_on_device} blocks on device, {self.blocks_to_swap} blocks on CPU")
            
            # Move first N blocks to GPU (these stay there)
            for i in range(num_blocks_on_device):
                if self.transformer_blocks[i].norm1.weight.device.type != self.device.type:
                    self.transformer_blocks[i].to(self.device)
            
            # Move last M blocks to CPU (these will be swapped via hooks)
            for i in range(num_blocks_on_device, self.num_blocks):
                if self.transformer_blocks[i].norm1.weight.device.type != "cpu":
                    self.transformer_blocks[i].to(torch.device("cpu"))
            
            if self.device.type == "cuda":
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
    
    def cleanup(self):
        """Remove all hooks and restore original methods."""
        for handle in self.forward_handles:
            handle.remove()
        for handle in self.backward_handles:
            handle.remove()
        
        # Restore monkey-patched methods
        for block, (method_name, original_method) in self.original_methods.items():
            setattr(block, method_name, original_method)
            
        self.forward_handles = []
        self.backward_handles = []
        self.original_methods = {}
    
    def __del__(self):
        """Cleanup hooks when manager is destroyed."""
        self.cleanup()


def enable_block_swap_for_sdxl_unet(
    unet: nn.Module,
    blocks_to_swap: int,
    device: torch.device,
    debug: bool = False
) -> SDXLBlockSwapManager:
    """
    Enable block swapping for SDXL UNet.
    
    Args:
        unet: The SDXL UNet model
        blocks_to_swap: Number of transformer blocks to swap between CPU and GPU
        device: The training device (usually CUDA)
        debug: Enable debug logging
    
    Returns:
        SDXLBlockSwapManager instance
    
    Example:
        >>> swap_manager = enable_block_swap_for_sdxl_unet(unet, blocks_to_swap=18, device=device)
        >>> # Before forward pass:
        >>> swap_manager.prepare_block_swap_before_forward()
        >>> # After training step:
        >>> swap_manager.wait_for_block_swap_to_finish()
    """
    manager = SDXLBlockSwapManager(unet, blocks_to_swap, device, debug)
    
    # Store the manager on the unet for easy access
    unet.block_swap_manager = manager
    
    return manager


def get_sdxl_block_info(unet: nn.Module) -> dict:
    """
    Get information about SDXL UNet blocks for debugging/configuration.
    
    Works with both:
    - Diffusers' UNet2DConditionModel (down_blocks, mid_block, up_blocks)
    - Kohya's SdxlUNet2DConditionModel (input_blocks, middle_block, output_blocks)
    
    Returns:
        Dictionary with block counts and memory information
    """
    transformer_blocks = collect_transformer_blocks_from_unet(unet)
    block_sizes = calculate_block_memory_sizes(transformer_blocks)
    
    total_memory = sum(block_sizes)
    avg_block_size = total_memory / len(transformer_blocks) if transformer_blocks else 0
    
    # Determine UNet type and count blocks per section
    # Kohya's custom SDXL UNet uses input_blocks, middle_block, output_blocks
    # Diffusers uses down_blocks, mid_block, up_blocks
    if hasattr(unet, 'input_blocks'):
        # Kohya's SdxlUNet2DConditionModel
        input_blocks_count = len(unet.input_blocks) if hasattr(unet, 'input_blocks') else 0
        middle_blocks_count = 1 if hasattr(unet, 'middle_block') else 0
        output_blocks_count = len(unet.output_blocks) if hasattr(unet, 'output_blocks') else 0
        
        return {
            'total_transformer_blocks': len(transformer_blocks),
            'down_blocks_transformer_count': input_blocks_count,  # Renamed for consistency
            'mid_blocks_transformer_count': middle_blocks_count,
            'up_blocks_transformer_count': output_blocks_count,   # Renamed for consistency
            'total_memory_gb': total_memory / (1024**3),
            'avg_block_size_mb': avg_block_size / (1024**2),
            'max_safely_swappable': len(transformer_blocks) - 2,
            'unet_type': 'kohya_sdxl',
        }
    else:
        # Diffusers' UNet2DConditionModel
        down_blocks_count = sum(1 for block in unet.down_blocks for _ in getattr(block, 'attentions', []))
        mid_blocks_count = 1 if hasattr(unet, 'mid_block') and hasattr(unet.mid_block, 'attentions') else 0
        up_blocks_count = sum(1 for block in unet.up_blocks for _ in getattr(block, 'attentions', []))
        
        return {
            'total_transformer_blocks': len(transformer_blocks),
            'down_blocks_transformer_count': down_blocks_count,
            'mid_blocks_transformer_count': mid_blocks_count,
            'up_blocks_transformer_count': up_blocks_count,
            'total_memory_gb': total_memory / (1024**3),
            'avg_block_size_mb': avg_block_size / (1024**2),
            'max_safely_swappable': len(transformer_blocks) - 2,
            'unet_type': 'diffusers',
        }

