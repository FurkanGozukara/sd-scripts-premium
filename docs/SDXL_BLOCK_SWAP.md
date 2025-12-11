# SDXL Block Swap (CPU Offloading)

## Overview

SDXL block swapping is a memory optimization technique that swaps transformer blocks between CPU and GPU memory during training. This allows training larger models or using larger batch sizes with limited VRAM.

## How It Works

Unlike FLUX which has uniform double/single blocks, SDXL has a non-uniform UNet structure:
- **Down blocks**: Various sizes of transformer blocks in the encoder
- **Mid block**: Transformer blocks in the middle of the UNet
- **Up blocks**: Various sizes of transformer blocks in the decoder
- **Total**: Typically ~70 transformer blocks (varies by SDXL variant)

The block swap implementation:
1. Identifies all `BasicTransformerBlock` instances in the UNet
2. Keeps the most frequently used blocks in VRAM
3. Swaps less frequently used blocks to CPU memory
4. Asynchronously transfers blocks between CPU and GPU during forward/backward passes

## Requirements

Block swapping for SDXL **requires** one of the following:
- `--fused_backward_pass`: Enables fused backward pass (recommended)
- `--blockwise_fused_optimizers`: Alternative approach using blockwise optimizers

**Note**: Block swapping is **not compatible** with gradient checkpointing for SDXL.

## Usage

### Command Line

```bash
# Basic usage with fused backward pass
python sdxl_train.py \
  --blocks_to_swap 30 \
  --fused_backward_pass \
  [other arguments...]

# With AdaFactor optimizer (recommended for fused backward pass)
python sdxl_train.py \
  --blocks_to_swap 30 \
  --fused_backward_pass \
  --optimizer_type adafactor \
  [other arguments...]
```

### Configuration File

```toml
[training]
blocks_to_swap = 30
fused_backward_pass = true
optimizer_type = "adafactor"
```

### GUI

In the SDXL-specific parameters section:
1. Enable **Fused backward pass**
2. Set **Transformer blocks to swap** slider (0-68)
   - 0 = Disabled (no block swapping)
   - 10-20 = Low swap (minimal VRAM savings, minimal speed impact)
   - 30-40 = Medium swap (moderate VRAM savings, moderate speed impact)
   - 50+ = High swap (maximum VRAM savings, significant speed impact)

## Recommendations

### VRAM Savings vs Speed Trade-off

| Blocks to Swap | VRAM Savings | Speed Impact | Use Case |
|----------------|--------------|--------------|----------|
| 0 | None | None | Default, no offloading |
| 10-20 | ~1-2 GB | Minimal (~5-10% slower) | Tight on VRAM, need speed |
| 30-40 | ~3-5 GB | Moderate (~20-30% slower) | Balanced approach |
| 50-60 | ~6-8 GB | Significant (~40-50% slower) | Maximum VRAM savings |

### Optimal Settings

**For 12GB VRAM (e.g., RTX 3060, RTX 4060 Ti):**
```bash
--blocks_to_swap 40
--fused_backward_pass
--train_batch_size 1
--gradient_accumulation_steps 4
```

**For 16GB VRAM (e.g., RTX 4060 Ti 16GB, Tesla T4):**
```bash
--blocks_to_swap 20
--fused_backward_pass
--train_batch_size 2
--gradient_accumulation_steps 2
```

**For 24GB VRAM (e.g., RTX 3090, RTX 4090):**
```bash
--blocks_to_swap 0  # Usually not needed
--train_batch_size 4
```

### Optimizer Considerations

- **AdaFactor**: Best compatibility with `--fused_backward_pass`
- **AdamW8bit**: Can work but may be slower
- **Other optimizers**: May not be compatible with fused backward pass

## Technical Details

### Block Collection

The implementation collects all `BasicTransformerBlock` instances from:
```
UNet2DConditionModel
├── down_blocks (CrossAttnDownBlock2D)
│   └── attentions (Transformer2DModel)
│       └── transformer_blocks (BasicTransformerBlock) ← Collected
├── mid_block (UNetMidBlock2DCrossAttn)
│   └── attentions (Transformer2DModel)
│       └── transformer_blocks (BasicTransformerBlock) ← Collected
└── up_blocks (CrossAttnUpBlock2D)
    └── attentions (Transformer2DModel)
        └── transformer_blocks (BasicTransformerBlock) ← Collected
```

### Memory Allocation

Unlike FLUX (uniform blocks), SDXL blocks have variable sizes:
- Blocks are tracked by actual memory footprint in bytes
- The swap strategy keeps the smallest N blocks in VRAM where N = total_blocks - blocks_to_swap
- Asynchronous transfer uses CUDA streams to overlap computation and memory transfers

### Comparison with OneTrainer

This implementation is inspired by OneTrainer's `LayerOffloadConductor` but adapted for Kohya's training pipeline:
- **OneTrainer**: Uses dynamic layer offloading with activation offloading
- **Kohya SDXL**: Uses block swapping with fused backward pass
- **Key difference**: We don't offload activations for SDXL due to skip connections

## Troubleshooting

### Error: "Block swapping requires fused_backward_pass"
**Solution**: Add `--fused_backward_pass` to your command line or enable it in the GUI.

### Error: "Requested X blocks to swap, but only Y can be safely swapped"
**Solution**: Reduce the `blocks_to_swap` value. The maximum safe value is typically total_blocks - 2.

### Training is much slower than expected
**Solution**: You may have set `blocks_to_swap` too high. Try reducing it by 20-30%.

### Out of VRAM even with block swapping
**Solutions**:
- Increase `blocks_to_swap` further
- Reduce `train_batch_size`
- Increase `gradient_accumulation_steps`
- Use a smaller resolution
- Enable `--cache_text_encoder_outputs`

### Gradient checkpointing not working
**Note**: Block swapping is incompatible with gradient checkpointing for SDXL. Choose one or the other.

## Performance Monitoring

To see block swap information during training:
```bash
python sdxl_train.py --blocks_to_swap 30 --fused_backward_pass --debug
```

This will show:
- Total transformer blocks found
- Memory per block
- Blocks kept on device vs swapped to CPU

## Related Options

- `--gradient_checkpointing`: NOT compatible with block swapping
- `--lowram`: Can be used together for additional memory savings during model loading
- `--cache_text_encoder_outputs`: Recommended to free up more VRAM
- `--xformers` or `--sdpa`: Memory-efficient attention, can be used together

## References

- Original implementation inspired by OneTrainer's `LayerOffloadConductor`
- See `library/sdxl_offloading_utils.py` for implementation details
- See `library/custom_offloading_utils.py` for base offloading utilities


