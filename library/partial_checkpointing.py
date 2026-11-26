"""Partial gradient checkpointing: ``--gradient_checkpointing_skip_blocks N`` keeps the activations of the last N blocks.

With gradient checkpointing every block recomputes its forward pass in the backward pass. The last N blocks in forward order
skip the recomputation instead, so spare VRAM buys speed: each skipped block costs the activation memory of one block and
saves the recomputation of one block. The blocks near the output are skipped because their activations are freed first in
the backward pass. 0 (the default) checkpoints every block. Gradients are the same as with full checkpointing.

The argument and its check are kohya-ss sd-scripts 7a4153d's ``library/args.py`` helpers
(``add_gradient_checkpointing_skip_blocks_argument``, ``verify_gradient_checkpointing_skip_blocks``) under kohya's names;
this branch has no ``library/args.py``. As in SECourses musubi-tuner (``training/partial_checkpointing.py``), the option
is refused with block swap: a block that keeps its activations keeps its weights' saved tensors for the backward pass.

Model contract (FLUX.1 ``flux_models.Flux``, SDXL ``sdxl_original_unet.SdxlUNet2DConditionModel``):
``checkpointed_block_count`` (the blocks counted, in forward order), ``set_gradient_checkpointing_skip_blocks(n)``
(validates ``n`` with ``validate_skip`` and stores it; applied at once when checkpointing is on, otherwise when it is
enabled) and ``enable_gradient_checkpointing(..., skip_blocks=None)``, where ``None`` keeps the stored value.
"""

from __future__ import annotations

import argparse
import logging

logger = logging.getLogger(__name__)

ARG = "gradient_checkpointing_skip_blocks"
OPTION = "--" + ARG


def add_gradient_checkpointing_skip_blocks_argument(parser: argparse.ArgumentParser):
    """Partial gradient checkpointing for the models whose `enable_gradient_checkpointing` accepts `skip_blocks`."""
    if OPTION in parser._option_string_actions:
        return
    parser.add_argument(
        OPTION,
        type=int,
        default=0,
        help="number of the last transformer blocks that keep their activations instead of recomputing them in the backward pass"
        " when --gradient_checkpointing is enabled. Each skipped block trades VRAM for speed. 0 (default) checkpoints all blocks."
        " Not with --blocks_to_swap."
        " / --gradient_checkpointing 有効時に、backward で再計算せず activation を保持する末尾の transformer ブロック数。"
        "スキップしたブロックの分だけ VRAM を使う代わりに速くなる。0（デフォルト）で全ブロックを checkpointing する",
    )


def requested_skip_blocks(args: argparse.Namespace) -> int:
    """The requested value as an int (a runtime TOML value reaches args without argparse's conversion)."""
    value = getattr(args, ARG, 0)
    if value in (None, "", False):
        return 0
    try:
        number = int(value)
    except (TypeError, ValueError):
        raise ValueError(f"{OPTION} must be a whole number, got {value!r}") from None
    try:
        whole = number == float(value)
    except (TypeError, ValueError):
        whole = False
    if not whole or isinstance(value, bool):
        raise ValueError(f"{OPTION} must be a whole number, got {value!r}")
    return number


def _swapped_blocks(args: argparse.Namespace) -> int:
    total = 0
    for key in ("blocks_to_swap", "double_blocks_to_swap", "single_blocks_to_swap"):  # flux_train.py keeps the deprecated two
        try:
            total += max(int(getattr(args, key, 0) or 0), 0)
        except (TypeError, ValueError):
            pass
    return total


def verify_gradient_checkpointing_skip_blocks(args: argparse.Namespace):
    """Validate --gradient_checkpointing_skip_blocks. It is ignored with a warning when gradient checkpointing is disabled,
    and refused with block swap. The upper bound is the model's block count (checked when the model gets the value)."""
    skip = requested_skip_blocks(args)
    if skip < 0:
        raise ValueError(f"{OPTION} must be 0 or positive, got {skip}")
    if skip > 0 and not getattr(args, "gradient_checkpointing", False):
        logger.warning(
            f"{OPTION} is ignored because --gradient_checkpointing is not enabled"
            f" / --gradient_checkpointing が無効なため {OPTION} は無視されます"
        )
        skip = 0
    if skip > 0 and _swapped_blocks(args) > 0:
        raise ValueError(
            f"{OPTION} cannot be combined with --blocks_to_swap: a block that skips checkpointing keeps its weights' saved"
            " tensors for the backward pass. Use block swap or skipped blocks, not both"
        )
    setattr(args, ARG, skip)


def block_is_checkpointed(index: int, total: int, skip: int) -> bool:
    """Whether block ``index`` (0-based, forward order) of ``total`` checkpointed blocks recomputes in the backward pass."""
    return index < total - skip


def validate_skip(skip, total: int, model_name: str) -> int:
    skip = int(skip or 0)
    if skip < 0 or skip > total:
        raise ValueError(f"{model_name}: {OPTION} must be 0 to {total} (the blocks it checkpoints), got {skip}")
    return skip


def apply(model, args: argparse.Namespace) -> int:
    """Hand the verified value to the model (``set_gradient_checkpointing_skip_blocks``). Returns it. A model without the
    contract is an error for a value above 0, never a silently ignored setting."""
    skip = requested_skip_blocks(args)
    setter = getattr(model, "set_gradient_checkpointing_skip_blocks", None)
    if not callable(setter):
        if skip > 0:
            raise ValueError(f"{OPTION} is not supported for {type(model).__name__}")
        return 0
    setter(skip)
    if skip > 0:
        logger.info(f"Partial gradient checkpointing: the last {skip} of {model.checkpointed_block_count} blocks keep their activations")
    return skip
