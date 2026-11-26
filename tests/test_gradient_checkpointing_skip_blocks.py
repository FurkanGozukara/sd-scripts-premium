"""--gradient_checkpointing_skip_blocks (partial gradient checkpointing) for FLUX.1 and SDXL.

Tiny random models on the CPU: the last N blocks in forward order keep their activations, the loss and every gradient
equal full checkpointing (and no checkpointing), the number of checkpoint calls drops by the skipped blocks' calls, and
the argument is validated as in kohya-ss sd-scripts 7a4153d (plus the block-swap refusal of SECourses musubi-tuner).
"""

import argparse
import logging
import sys

import pytest
import torch
import torch.utils.checkpoint

from library import flux_models, partial_checkpointing, sdxl_original_unet

# --------------------------------------------------------------------------------------------------------------------
# FLUX.1


def _tiny_flux() -> flux_models.Flux:
    torch.manual_seed(0)
    params = flux_models.FluxParams(
        in_channels=16,
        vec_in_dim=24,
        context_in_dim=20,
        hidden_size=64,
        mlp_ratio=2.0,
        num_heads=2,
        depth=2,
        depth_single_blocks=3,
        axes_dim=[8, 12, 12],
        theta=10_000,
        qkv_bias=True,
        guidance_embed=True,
    )
    model = flux_models.Flux(params).float()
    for parameter in model.parameters():  # the final layer etc. are not zero-initialised here, but make every grad non-trivial
        torch.nn.init.normal_(parameter, std=0.05)
    return model


def _flux_inputs(device="cpu"):
    generator = torch.Generator().manual_seed(1)
    batch, img_len, txt_len = 2, 12, 5
    img = torch.randn(batch, img_len, 16, generator=generator)
    img_ids = torch.zeros(batch, img_len, 3)
    img_ids[..., 1] = torch.arange(img_len).float()[None] // 4
    img_ids[..., 2] = torch.arange(img_len).float()[None] % 4
    txt = torch.randn(batch, txt_len, 20, generator=generator)
    txt_ids = torch.zeros(batch, txt_len, 3)
    y = torch.randn(batch, 24, generator=generator)
    timesteps = torch.tensor([0.3, 0.8])
    guidance = torch.tensor([1.0, 3.5])
    mask = torch.ones(batch, txt_len)
    mask[1, -2:] = 0
    tensors = dict(img=img, img_ids=img_ids, txt=txt, txt_ids=txt_ids, y=y, timesteps=timesteps, guidance=guidance, txt_attention_mask=mask)
    return {key: value.to(device) for key, value in tensors.items()}


def _flux_step(model, inputs, skip=None, checkpointing=True, cpu_offload=False):
    model.train()
    model.zero_grad(set_to_none=True)
    if checkpointing:
        model.enable_gradient_checkpointing(cpu_offload=cpu_offload, skip_blocks=skip)
    else:
        model.disable_gradient_checkpointing()
    img = inputs["img"].clone().requires_grad_(True)  # as the trainers do with checkpointing
    out = model(**{**inputs, "img": img})
    loss = (out.float() ** 2).mean()
    loss.backward()
    grads = {name: parameter.grad.detach().clone() for name, parameter in model.named_parameters() if parameter.grad is not None}
    return loss.detach(), grads, img.grad.detach().clone()


def _assert_same(reference, other):
    ref_loss, ref_grads, ref_input = reference
    loss, grads, input_grad = other
    torch.testing.assert_close(loss, ref_loss, rtol=1e-6, atol=1e-7)
    assert grads.keys() == ref_grads.keys()
    for name in ref_grads:
        torch.testing.assert_close(grads[name], ref_grads[name], rtol=1e-5, atol=1e-7, msg=name)
    torch.testing.assert_close(input_grad, ref_input, rtol=1e-5, atol=1e-7)


def _flux_flags(model):
    return [block.gradient_checkpointing for block in list(model.double_blocks) + list(model.single_blocks)]


def test_flux_block_count_in_forward_order():
    assert _tiny_flux().checkpointed_block_count == 5
    with torch.device("meta"):
        dev = flux_models.Flux(flux_models.configs["dev"].params)
    assert dev.checkpointed_block_count == 57  # 19 double + 38 single stream blocks


def test_flux_skip_keeps_gradients_and_drops_checkpoint_calls(monkeypatch):
    model = _tiny_flux()
    inputs = _flux_inputs()
    calls = []
    real_checkpoint = flux_models.checkpoint

    def counting_checkpoint(function, *args, **kwargs):
        calls.append(function)
        return real_checkpoint(function, *args, **kwargs)

    monkeypatch.setattr(flux_models, "checkpoint", counting_checkpoint)

    plain = _flux_step(model, inputs, checkpointing=False)
    assert not calls
    results = {}
    for skip in (0, 2, 5):
        calls.clear()
        results[skip] = _flux_step(model, inputs, skip=skip)
        assert _flux_flags(model) == [True] * (5 - skip) + [False] * skip
        # time / vector / guidance embedders (3) are always checkpointed, the blocks only up to the skipped ones
        assert len(calls) == 3 + 5 - skip
    for skip in (0, 2, 5):
        _assert_same(plain, results[skip])


def test_flux_skip_spans_double_and_single_blocks():
    model = _tiny_flux()
    model.enable_gradient_checkpointing(skip_blocks=4)  # all 3 single blocks and the last double block
    assert _flux_flags(model) == [True, False, False, False, False]
    model.enable_gradient_checkpointing(skip_blocks=1)
    assert _flux_flags(model) == [True, True, True, True, False]


def test_flux_stored_value_survives_a_plain_enable():
    # LoRA path: the trainer stores the value after loading, train_network then calls enable_gradient_checkpointing()
    model = _tiny_flux()
    model.set_gradient_checkpointing_skip_blocks(2)
    assert not model.gradient_checkpointing and not any(_flux_flags(model))
    model.enable_gradient_checkpointing()
    assert _flux_flags(model) == [True, True, True, False, False]
    model.enable_gradient_checkpointing(cpu_offload=False)
    assert _flux_flags(model) == [True, True, True, False, False]
    # changed while enabled: applied at once; back to 0 checkpoints every block again
    model.set_gradient_checkpointing_skip_blocks(3)
    assert _flux_flags(model) == [True, True, False, False, False]
    model.set_gradient_checkpointing_skip_blocks(0)
    assert all(_flux_flags(model))
    model.disable_gradient_checkpointing()
    assert not any(_flux_flags(model))


@pytest.mark.parametrize("skip", [-1, 6])
def test_flux_out_of_range_is_refused(skip):
    model = _tiny_flux()
    with pytest.raises(ValueError, match="0 to 5"):
        model.set_gradient_checkpointing_skip_blocks(skip)
    with pytest.raises(ValueError, match="0 to 5"):
        model.enable_gradient_checkpointing(skip_blocks=skip)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CPU-offloaded checkpoints need CUDA")
def test_flux_skip_with_cpu_offload_checkpointing_on_cuda():
    model = _tiny_flux().cuda()
    inputs = _flux_inputs("cuda")
    reference = _flux_step(model, inputs, skip=0, cpu_offload=True)
    plain = _flux_step(model, inputs, skip=0, cpu_offload=False)
    for skip in (2, 4, 5):  # the CPU -> GPU boundary inside the double blocks, between the lists, and none
        _assert_same(reference, _flux_step(model, inputs, skip=skip, cpu_offload=True))
    _assert_same(reference, plain)


# --------------------------------------------------------------------------------------------------------------------
# SDXL


@pytest.fixture
def tiny_sdxl(monkeypatch):
    # the real layout (9 input blocks, middle block, 9 output blocks, 10-layer transformers at level 2) at small widths
    monkeypatch.setattr(sdxl_original_unet, "MODEL_CHANNELS", 32)
    monkeypatch.setattr(sdxl_original_unet, "TIME_EMBED_DIM", 64)
    monkeypatch.setattr(sdxl_original_unet, "ADM_IN_CHANNELS", 48)
    torch.manual_seed(0)
    unet = sdxl_original_unet.SdxlUNet2DConditionModel().float()
    for parameter in unet.parameters():  # zero-initialised output convs would leave most gradients zero
        torch.nn.init.normal_(parameter, std=0.02)
    return unet


CHECKPOINTED_TYPES = (
    sdxl_original_unet.ResnetBlock2D,
    sdxl_original_unet.Downsample2D,
    sdxl_original_unet.Upsample2D,
    sdxl_original_unet.BasicTransformerBlock,
)


def _sdxl_inputs():
    generator = torch.Generator().manual_seed(2)
    x = torch.randn(2, 4, 16, 16, generator=generator)
    timesteps = torch.tensor([100, 700])
    context = torch.randn(2, 6, 2048, generator=generator)
    y = torch.randn(2, 48, generator=generator)
    return x, timesteps, context, y


def _sdxl_step(unet, inputs, skip=None, checkpointing=True):
    unet.train()
    unet.zero_grad(set_to_none=True)
    if checkpointing:
        unet.enable_gradient_checkpointing(skip_blocks=skip)
    else:
        unet.disable_gradient_checkpointing()
    x, timesteps, context, y = inputs
    x = x.clone().requires_grad_(True)  # train_network.py: the hidden state requires grad with checkpointing
    out = unet(x, timesteps, context, y)
    loss = (out.float() ** 2).mean()
    loss.backward()
    grads = {name: parameter.grad.detach().clone() for name, parameter in unet.named_parameters() if parameter.grad is not None}
    return loss.detach(), grads, x.grad.detach().clone()


def _sdxl_block_flags(unet):
    return [
        any(getattr(module, "gradient_checkpointing", False) for module in block.modules()) for block in unet._gradient_checkpointing_blocks()
    ]


def test_sdxl_block_count_in_forward_order(tiny_sdxl):
    blocks = tiny_sdxl._gradient_checkpointing_blocks()
    assert tiny_sdxl.checkpointed_block_count == len(blocks) == 18
    assert blocks[:8] == list(tiny_sdxl.input_blocks)[1:]  # input block 0 is conv_in, never checkpointed
    assert blocks[8] is tiny_sdxl.middle_block
    assert blocks[9:] == list(tiny_sdxl.output_blocks)


def test_sdxl_real_size_block_count():
    with torch.device("meta"):
        unet = sdxl_original_unet.SdxlUNet2DConditionModel()
    assert unet.checkpointed_block_count == 18


def test_sdxl_skip_keeps_gradients_and_drops_checkpoint_calls(tiny_sdxl, monkeypatch):
    unet = tiny_sdxl
    inputs = _sdxl_inputs()
    calls = []
    real_checkpoint = torch.utils.checkpoint.checkpoint

    def counting_checkpoint(function, *args, **kwargs):
        calls.append(function)
        return real_checkpoint(function, *args, **kwargs)

    monkeypatch.setattr(torch.utils.checkpoint, "checkpoint", counting_checkpoint)
    per_block = [sum(isinstance(module, CHECKPOINTED_TYPES) for module in block.modules()) for block in unet._gradient_checkpointing_blocks()]
    assert all(count > 0 for count in per_block)

    plain = _sdxl_step(unet, inputs, checkpointing=False)
    assert not calls
    results = {}
    for skip in (0, 3, 9, 18):  # level-0 output ResNets; all output blocks; everything
        calls.clear()
        results[skip] = _sdxl_step(unet, inputs, skip=skip)
        assert _sdxl_block_flags(unet) == [True] * (18 - skip) + [False] * skip
        assert len(calls) == sum(per_block[: 18 - skip])
    assert len(calls) == 0
    for skip in (0, 3, 9, 18):
        _assert_same(plain, results[skip])


def test_sdxl_stored_value_survives_a_plain_enable(tiny_sdxl):
    unet = tiny_sdxl
    unet.set_gradient_checkpointing_skip_blocks(10)  # every output block and the middle block
    assert not any(_sdxl_block_flags(unet))
    unet.enable_gradient_checkpointing()  # train_network.py's call
    assert _sdxl_block_flags(unet) == [True] * 8 + [False] * 10
    unet.set_gradient_checkpointing_skip_blocks(0)
    assert all(_sdxl_block_flags(unet))
    unet.disable_gradient_checkpointing()
    assert not any(_sdxl_block_flags(unet))
    with pytest.raises(ValueError, match="0 to 18"):
        unet.set_gradient_checkpointing_skip_blocks(19)
    with pytest.raises(ValueError, match="0 to 18"):
        unet.enable_gradient_checkpointing(skip_blocks=-1)


# --------------------------------------------------------------------------------------------------------------------
# argument


def _args(**values):
    base = dict(gradient_checkpointing=True, gradient_checkpointing_skip_blocks=0, blocks_to_swap=None)
    base.update(values)
    return argparse.Namespace(**base)


def test_verify_accepts_and_normalises():
    for value, expected in ((0, 0), (7, 7), ("12", 12), (12.0, 12), (None, 0), ("", 0)):
        args = _args(gradient_checkpointing_skip_blocks=value)
        partial_checkpointing.verify_gradient_checkpointing_skip_blocks(args)
        assert args.gradient_checkpointing_skip_blocks == expected
    args = argparse.Namespace(gradient_checkpointing=True)  # a parser without the option (other trainers)
    partial_checkpointing.verify_gradient_checkpointing_skip_blocks(args)
    assert args.gradient_checkpointing_skip_blocks == 0


@pytest.mark.parametrize("value", [-1, 2.5, "two", True])
def test_verify_refuses_bad_values(value):
    with pytest.raises(ValueError, match="gradient_checkpointing_skip_blocks"):
        partial_checkpointing.verify_gradient_checkpointing_skip_blocks(_args(gradient_checkpointing_skip_blocks=value))


def test_verify_ignores_the_value_without_checkpointing(caplog):
    args = _args(gradient_checkpointing=False, gradient_checkpointing_skip_blocks=8, blocks_to_swap=10)
    with caplog.at_level(logging.WARNING, logger=partial_checkpointing.__name__):
        partial_checkpointing.verify_gradient_checkpointing_skip_blocks(args)
    assert args.gradient_checkpointing_skip_blocks == 0
    assert "ignored" in caplog.text


@pytest.mark.parametrize("swap", [dict(blocks_to_swap=4), dict(double_blocks_to_swap=2), dict(single_blocks_to_swap=4)])
def test_verify_refuses_block_swap(swap):
    with pytest.raises(ValueError, match="blocks_to_swap"):
        partial_checkpointing.verify_gradient_checkpointing_skip_blocks(_args(gradient_checkpointing_skip_blocks=3, **swap))
    args = _args(gradient_checkpointing_skip_blocks=0, **swap)  # 0 with block swap stays allowed
    partial_checkpointing.verify_gradient_checkpointing_skip_blocks(args)


def test_apply_refuses_a_model_without_the_contract():
    with pytest.raises(ValueError, match="not supported for Linear"):
        partial_checkpointing.apply(torch.nn.Linear(2, 2), _args(gradient_checkpointing_skip_blocks=1))
    assert partial_checkpointing.apply(torch.nn.Linear(2, 2), _args()) == 0
    model = _tiny_flux()
    assert partial_checkpointing.apply(model, _args(gradient_checkpointing_skip_blocks=3)) == 3
    assert model.gradient_checkpointing_skip_blocks == 3


def _parsers():
    import flux_train
    import flux_train_network
    import sdxl_train
    import sdxl_train_network

    return {
        "flux_train_network": flux_train_network.setup_parser(),
        "flux_train": flux_train.setup_parser(),
        "sdxl_train_network": sdxl_train_network.setup_parser(),
        "sdxl_train": sdxl_train.setup_parser(),
    }


def _dataset_group():
    from unittest import mock

    group = mock.MagicMock()
    group.is_text_encoder_output_cacheable.return_value = True
    return group


def test_flux_lora_trainer_checks_the_option_before_loading():
    import flux_train_network

    parser = flux_train_network.setup_parser()
    trainer = flux_train_network.FluxNetworkTrainer()
    args = parser.parse_args(["--gradient_checkpointing", "--gradient_checkpointing_skip_blocks", "20"])
    trainer.assert_extra_args(args, _dataset_group(), None)
    assert args.gradient_checkpointing_skip_blocks == 20
    # --split_mode turns on block swap 18 inside assert_extra_args: the check runs after it
    args = parser.parse_args(["--gradient_checkpointing", "--gradient_checkpointing_skip_blocks", "20", "--split_mode"])
    with pytest.raises(ValueError, match="blocks_to_swap"):
        flux_train_network.FluxNetworkTrainer().assert_extra_args(args, _dataset_group(), None)
    args = parser.parse_args(["--gradient_checkpointing", "--gradient_checkpointing_skip_blocks", "2", "--model_type", "chroma", "--apply_t5_attn_mask"])
    with pytest.raises(ValueError, match="Chroma"):
        flux_train_network.FluxNetworkTrainer().assert_extra_args(args, _dataset_group(), None)
    args = parser.parse_args(["--gradient_checkpointing_skip_blocks", "20"])  # no checkpointing: ignored
    flux_train_network.FluxNetworkTrainer().assert_extra_args(args, _dataset_group(), None)
    assert args.gradient_checkpointing_skip_blocks == 0


def test_sdxl_lora_trainer_checks_the_option_before_loading():
    import sdxl_train_network

    parser = sdxl_train_network.setup_parser()
    args = parser.parse_args(["--gradient_checkpointing", "--gradient_checkpointing_skip_blocks", "9"])
    sdxl_train_network.SdxlNetworkTrainer().assert_extra_args(args, _dataset_group(), None)
    assert args.gradient_checkpointing_skip_blocks == 9
    args.blocks_to_swap = 4  # SDXL takes blocks_to_swap from the runtime TOML only (no argparse option)
    with pytest.raises(ValueError, match="blocks_to_swap"):
        sdxl_train_network.SdxlNetworkTrainer().assert_extra_args(args, _dataset_group(), None)


def test_trainers_take_the_option_and_the_runtime_toml_key(tmp_path, monkeypatch):
    from library import train_util

    config = tmp_path / "run.toml"
    config.write_text("gradient_checkpointing = true\ngradient_checkpointing_skip_blocks = 21\n", encoding="utf-8")
    for name, parser in _parsers().items():
        assert parser.parse_args([]).gradient_checkpointing_skip_blocks == 0, name
        assert parser.parse_args(["--gradient_checkpointing_skip_blocks", "9"]).gradient_checkpointing_skip_blocks == 9, name
        # read_config_from_file parses the command line again on top of the TOML, as the trainers' __main__ does
        monkeypatch.setattr(sys, "argv", [f"{name}.py", "--config_file", str(config)])
        args = train_util.read_config_from_file(parser.parse_args(sys.argv[1:]), parser)
        assert args.gradient_checkpointing_skip_blocks == 21, name
        assert args.gradient_checkpointing is True, name
