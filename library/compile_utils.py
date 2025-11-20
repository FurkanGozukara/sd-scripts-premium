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
) -> nn.Module:
    """
    Compile model blocks using torch.compile.
    
    Args:
        args: Namespace containing compile arguments
        model: The model to compile
        target_blocks: List of block lists to compile (e.g., [unet.down_blocks, unet.up_blocks])
        disable_linear: If True, disable compile for Linear layers (useful with block swapping)
        log_prefix: Prefix for log messages
        
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
        for blocks in target_blocks:
            if blocks is None:
                continue
            for block in blocks:
                disable_linear_from_compile(block)
    
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
    for blocks in target_blocks:
        if blocks is None:
            continue
            
        # Handle both ModuleList and regular lists
        for i, block in enumerate(blocks):
            compiled_block = torch.compile(
                block,
                backend=args.compile_backend,
                mode=args.compile_mode,
                dynamic=compile_dynamic,
                fullgraph=args.compile_fullgraph,
            )
            blocks[i] = compiled_block
    
    return model


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

