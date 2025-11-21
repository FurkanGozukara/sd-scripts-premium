"""
Utilities for torch.compile support in training scripts.
Based on musubi-tuner implementation with adaptations for sd-scripts.
"""

import argparse
import logging
from typing import List, Optional, Union

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def add_compile_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """
    Add torch.compile related arguments to parser.
    
    Args:
        parser: ArgumentParser to add arguments to
        
    Returns:
        parser with compile arguments added
    """
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Enable torch.compile for model blocks (requires PyTorch 2.1+, Triton for CUDA) / モデルブロックにtorch.compileを有効化（PyTorch 2.1+、CUDAの場合はTritonが必要）",
    )
    parser.add_argument(
        "--compile_backend",
        type=str,
        default="inductor",
        help="torch.compile backend (default: inductor) / torch.compileのバックエンド（デフォルト: inductor）",
    )
    parser.add_argument(
        "--compile_mode",
        type=str,
        default="default",
        choices=["default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs"],
        help="torch.compile mode (default: default) / torch.compileのモード（デフォルト: default）",
    )
    parser.add_argument(
        "--compile_dynamic",
        type=str,
        default=None,
        choices=["true", "false", "auto"],
        help="Dynamic shapes mode for torch.compile (default: None, same as auto). "
        "True enables dynamic shapes, False disables them, auto/None lets PyTorch decide / "
        "torch.compileの動的形状モード（デフォルト: None、autoと同じ動作）",
    )
    parser.add_argument(
        "--compile_fullgraph",
        action="store_true",
        help="Enable fullgraph mode in torch.compile / torch.compileでフルグラフモードを有効にする",
    )
    parser.add_argument(
        "--compile_cache_size_limit",
        type=int,
        default=None,
        help="Set torch._dynamo.config.cache_size_limit (default: PyTorch default, typically 8-32) / "
        "torch._dynamo.config.cache_size_limitを設定（デフォルト: PyTorchのデフォルト、通常8-32）",
    )
    
    return parser


def disable_linear_from_compile(module: nn.Module):
    """
    Monkey-patch to disable torch.compile for all Linear layers in the given module.
    This is useful when using block swapping or CPU offloading features.
    
    Based on musubi-tuner implementation.
    
    Args:
        module: Module to disable compile for linear layers
    """
    for sub_module in module.modules():
        # Check if the class name ends with 'Linear' (handles nn.Linear and custom Linear layers)
        if sub_module.__class__.__name__.endswith("Linear"):
            if not hasattr(sub_module, "_forward_before_disable_compile"):
                sub_module._forward_before_disable_compile = sub_module.forward
                sub_module._eager_forward = torch._dynamo.disable()(sub_module.forward)
            sub_module.forward = sub_module._eager_forward  # override forward to disable compile


def compile_model(
    args: argparse.Namespace,
    model: nn.Module,
    target_blocks: List[Union[nn.ModuleList, List[nn.Module]]],
    disable_linear: bool = False,
    log_prefix: str = "Model",
    skip_blocks: Optional[int] = None,
    skip_first_blocks_per_list: Optional[List[Optional[int]]] = None,
    skip_last_blocks_per_list: Optional[List[Optional[int]]] = None,
) -> nn.Module:
    """
    Compile model blocks using torch.compile.
    
    Args:
        args: Namespace containing compile arguments
        model: The model to compile
        target_blocks: List of block lists to compile (e.g., [unet.down_blocks, unet.up_blocks])
        disable_linear: If True, disable compile for Linear layers (useful with block swapping)
        log_prefix: Prefix for log messages
        skip_blocks: If specified, skip compiling the first N blocks (useful with block swapping)
        skip_first_blocks_per_list: Optional list matching target_blocks specifying how many leading
            blocks to skip compiling for each block list (useful when those blocks are swapped/offloaded)
        skip_last_blocks_per_list: Optional list matching target_blocks specifying how many trailing
            blocks to skip compiling for each block list (useful when those blocks are swapped/offloaded)
        
    Returns:
        The model with compiled blocks
    """
    if not hasattr(args, 'compile') or not args.compile:
        return model
        
    # Check PyTorch version
    torch_version = tuple(int(x) for x in torch.__version__.split('.')[:2])
    if torch_version < (2, 1):
        logger.warning(
            f"torch.compile requires PyTorch 2.1+, but found {torch.__version__}. Skipping compilation."
        )
        return model
    
    if disable_linear:
        logger.info(f"Disabling torch.compile for Linear layers in {log_prefix} (due to block swapping/offloading)...")
        linear_count = 0
        for blocks in target_blocks:
            if blocks is None:
                continue
            for block in blocks:
                before_count = sum(1 for m in block.modules() if m.__class__.__name__.endswith("Linear"))
                disable_linear_from_compile(block)
                linear_count += before_count
        logger.info(f"Disabled torch.compile for {linear_count} Linear layers across all blocks")
    
    # Convert compile_dynamic string to boolean or None
    compile_dynamic = None
    if hasattr(args, 'compile_dynamic') and args.compile_dynamic is not None:
        compile_dynamic = {"true": True, "false": False, "auto": None}[args.compile_dynamic.lower()]
    
    logger.info(
        f"Compiling {log_prefix} with torch.compile: "
        f"backend={args.compile_backend}, mode={args.compile_mode}, "
        f"dynamic={compile_dynamic}, fullgraph={args.compile_fullgraph}"
    )
    
    # Set cache size limit if specified
    if hasattr(args, 'compile_cache_size_limit') and args.compile_cache_size_limit is not None:
        torch._dynamo.config.cache_size_limit = args.compile_cache_size_limit
        logger.info(f"Set torch._dynamo.config.cache_size_limit to {args.compile_cache_size_limit}")
    
    # Compile each block in target_blocks
    blocks_compiled = 0
    blocks_skipped = 0
    
    logger.info(f"{log_prefix}: Starting compilation of {sum(len(b) for b in target_blocks if b is not None)} blocks...")
    
    global_skip_first = skip_blocks or 0

    for block_list_idx, blocks in enumerate(target_blocks):
        if blocks is None or len(blocks) == 0:
            continue
        
        skip_first = global_skip_first
        if skip_first_blocks_per_list is not None and block_list_idx < len(skip_first_blocks_per_list):
            per_list_value = skip_first_blocks_per_list[block_list_idx]
            if per_list_value is not None and per_list_value > 0:
                skip_first += per_list_value
        
        skip_last = 0
        if skip_last_blocks_per_list is not None and block_list_idx < len(skip_last_blocks_per_list):
            per_list_last = skip_last_blocks_per_list[block_list_idx]
            if per_list_last is not None and per_list_last > 0:
                skip_last = min(len(blocks), max(0, per_list_last))
        
        # Handle both ModuleList and regular lists
        for i, block in enumerate(blocks):
            # Skip first N blocks if skip_blocks specified (these are swapped to CPU)
            if skip_first is not None and i < skip_first:
                blocks_skipped += 1
                continue
            if skip_last > 0 and (len(blocks) - i) <= skip_last:
                blocks_skipped += 1
                continue
            
            logger.debug(f"{log_prefix}: Compiling block_list[{block_list_idx}][{i}] - {block.__class__.__name__}")
            
            compiled_block = torch.compile(
                block,
                backend=args.compile_backend,
                mode=args.compile_mode,
                dynamic=compile_dynamic,
                fullgraph=args.compile_fullgraph,
            )
            blocks[i] = compiled_block
            blocks_compiled += 1
            
            # Verify compilation worked
            if hasattr(compiled_block, '_torchdynamo_orig_callable'):
                logger.debug(f"  ✓ Block {i} successfully wrapped by torch.compile")
            else:
                logger.warning(f"  ✗ Block {i} may not be properly compiled!")
    
    if blocks_skipped > 0:
        logger.info(f"Compiled {blocks_compiled} blocks, skipped {blocks_skipped} blocks (swap/offloaded)")
    else:
        logger.info(f"Compiled {blocks_compiled} blocks")
    
    logger.info(
        f"{log_prefix}: ⚡ Compilation setup complete. "
        f"IMPORTANT: Speedup will appear after 10-20 steps (during first forward passes, graphs are captured and compiled). "
        f"First epoch will be SLOWER. Compare epoch 2+ for speedup measurement."
    )
    
    return model


def compile_flux_with_block_swap(
    args: argparse.Namespace,
    flux_model: nn.Module,
    blocks_to_swap: int,
    log_prefix: str = "FLUX",
) -> nn.Module:
    """
    Intelligently compile FLUX model considering block swapping.

    When block swapping is enabled, we use fullgraph=True for better compatibility
    with dynamic weight movement between CPU and GPU.

    Args:
        args: Namespace containing compile arguments
        flux_model: The FLUX model to compile
        blocks_to_swap: Number of blocks being swapped (from args.blocks_to_swap)
        log_prefix: Prefix for log messages

    Returns:
        The model with compiled blocks
    """
    if not hasattr(args, 'compile') or not args.compile:
        return flux_model

    # Check PyTorch version
    torch_version = tuple(int(x) for x in torch.__version__.split('.')[:2])
    if torch_version < (2, 1):
        logger.warning(
            f"torch.compile requires PyTorch 2.1+, but found {torch.__version__}. Skipping compilation."
        )
        return flux_model

    unwrapped = flux_model
    if hasattr(flux_model, 'module'):
        unwrapped = flux_model.module

    # When block swapping is enabled, use fullgraph=True for better compatibility
    # with dynamic device movement, and don't disable Linear layer compilation
    skip_first_config: Optional[List[Optional[int]]] = None
    skip_last_config: Optional[List[Optional[int]]] = None

    if blocks_to_swap > 0:
        logger.info(
            f"{log_prefix}: Block swap enabled ({blocks_to_swap} blocks). "
            f"Enabling fullgraph mode for better compatibility with block swapping."
        )
        # Temporarily override compile_fullgraph
        original_fullgraph = getattr(args, 'compile_fullgraph', False)
        args.compile_fullgraph = True
        disable_linear = False

        double_blocks = getattr(unwrapped, "double_blocks", None)
        single_blocks = getattr(unwrapped, "single_blocks", None)

        total_double = len(double_blocks) if double_blocks is not None else 0
        total_single = len(single_blocks) if single_blocks is not None else 0

        double_blocks_to_swap = min(blocks_to_swap // 2, total_double)
        remaining_for_single = blocks_to_swap - double_blocks_to_swap
        single_blocks_to_swap = min(remaining_for_single * 2, total_single)

        non_swapped_double = total_double - double_blocks_to_swap
        non_swapped_single = total_single - single_blocks_to_swap

        logger.info(
            f"{log_prefix}: Selectively compiling non-swapped blocks "
            f"(double={non_swapped_double}/{total_double}, single={non_swapped_single}/{total_single})."
        )

        skip_sizes = [
            double_blocks_to_swap if total_double > 0 else 0,
            single_blocks_to_swap if total_single > 0 else 0,
        ]
        skip_first_config = skip_sizes
        skip_last_config = skip_sizes
    else:
        logger.info(f"{log_prefix}: Block swap disabled. Using standard compilation.")
        disable_linear = False

    target_blocks = [unwrapped.double_blocks, unwrapped.single_blocks]

    try:
        return compile_model(
            args,
            flux_model,
            target_blocks,
            disable_linear=disable_linear,
            log_prefix=log_prefix,
            skip_first_blocks_per_list=skip_first_config,
            skip_last_blocks_per_list=skip_last_config,
        )
    finally:
        # Restore original fullgraph setting
        if blocks_to_swap > 0:
            args.compile_fullgraph = original_fullgraph


def maybe_uncompile_state_dict(state_dict: dict) -> dict:
    """
    Remove '_orig_mod.' prefixes from state dict keys that appear when saving compiled models.
    
    Args:
        state_dict: State dict potentially containing compiled model keys
        
    Returns:
        State dict with '_orig_mod.' prefixes removed
    """
    # Check if this is a compiled model by looking for _orig_mod in keys
    has_orig_mod = any("_orig_mod." in key for key in state_dict.keys())
    
    if has_orig_mod:
        logger.info("Detected compiled model, removing '_orig_mod.' prefixes from state dict for saving")
        new_state_dict = {}
        for key, value in state_dict.items():
            new_key = key.replace("_orig_mod.", "")
            new_state_dict[new_key] = value
        return new_state_dict
    
    return state_dict

