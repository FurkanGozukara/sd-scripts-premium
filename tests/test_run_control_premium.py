"""Premium-only parts of run control (kohya-ss sd-scripts 0.13.0 port): pause keeps any block-swap placement and parks
the H2D-only ring, the LoRA scripts' parser has --run_control, and the premium samplers (train_util.sample_images_common
for SD / SDXL, flux_train_utils.sample_images for FLUX.1) follow the schedule, the reserved prompts and a command's
forced prompts through library/run_control_sampling.py."""

import argparse
from types import SimpleNamespace

import pytest
import torch

from library import run_control
from library import run_control_sampling as sampling


class _StreamOffloader:
    def __init__(self, block):
        self.block = block
        self.ring_param = ["slot 0", "slot 1"]
        self.cpu_master = {3: torch.nn.Parameter(torch.ones(2), requires_grad=False)}
        self.in_slot = [3, None]
        self.free_event = ["event", None]
        self.calls = []
        self.copier = SimpleNamespace(sync=lambda: self.calls.append("sync"), reset=lambda: self.calls.append("reset"))

    def _bind(self, block_idx, tensors):
        self.calls.append(("bind", block_idx))
        self.block.weight = tensors


def test_pause_parks_the_h2d_ring_and_keeps_cpu_tensors():
    model = torch.nn.Linear(2, 2)
    model.offloader = _StreamOffloader(model)
    master = model.offloader.cpu_master[3]
    restore = run_control.offload_modules_for_pause([model, None])
    assert model.offloader.calls == ["sync", ("bind", 3), "reset"]
    assert model.weight is master and model.offloader.in_slot == [None, None]
    assert restore == []  # nothing was on an accelerator
    run_control.restore_modules_after_pause(restore)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="moves tensors between CUDA and the CPU")
def test_pause_returns_each_tensor_to_its_device():
    model = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.Linear(4, 4)).cuda()
    model[1].to("cpu")  # a block the classic block swap keeps on the CPU
    weights = [model[0].weight, model[1].weight]
    restore = run_control.offload_modules_for_pause([model])
    assert all(p.device.type == "cpu" for p in model.parameters())
    run_control.restore_modules_after_pause(restore)
    assert model[0].weight.device.type == "cuda" and model[1].weight.device.type == "cpu"
    assert model[0].weight is weights[0] and model[1].weight is weights[1]


def test_the_lora_scripts_parse_run_control():
    import train_network

    args = train_network.setup_parser().parse_args(["--run_control"])
    assert args.run_control is True
    assert train_network.setup_parser().parse_args([]).run_control is False


@pytest.fixture
def prompt_file(tmp_path):
    path = tmp_path / "prompts.txt"
    path.write_text("a cat --s 20 --d 1\na reserved prompt --s 0\na dog --d 2\n", encoding="utf-8")
    return str(path)


def _args(prompt_file, **kwargs):
    values = dict(sample_prompts=prompt_file, sample_at_first=False, sample_every_n_steps=None, sample_every_n_epochs=None,
                  output_dir="out", output_name="x")
    values.update(kwargs)
    return argparse.Namespace(**values)


def test_premium_samplers_use_the_shared_decision_and_prompts(prompt_file, monkeypatch):
    from library import flux_train_utils, train_util

    seen = []
    monkeypatch.setattr(train_util, "PartialState", lambda: (_ for _ in ()).throw(RuntimeError("sampling started")), raising=False)
    monkeypatch.setattr(flux_train_utils, "PartialState", lambda: (_ for _ in ()).throw(RuntimeError("sampling started")))

    def run(fn):
        try:
            fn()
        except RuntimeError as e:
            seen.append(str(e))
            return True
        return False

    args = _args(prompt_file, sample_every_n_steps=10)
    flux = lambda steps, epoch=None: flux_train_utils.sample_images(None, args, epoch, steps, None, None, None, None)  # noqa: E731
    assert not run(lambda: flux(5))  # not due
    assert run(lambda: flux(10))  # due by the schedule
    with sampling.force_sample_prompts(sampling.build_forced_prompts(args, indices=[1])):
        assert run(lambda: flux(7))  # a command: any step
    sdxl = lambda steps: train_util.sample_images_common(None, None, args, None, steps, None, None, None, None, None)  # noqa: E731
    assert not run(lambda: sdxl(5))
    with sampling.force_sample_prompts([{"prompt": "a cat", "enum": 100}]):
        assert run(lambda: sdxl(5))
    # the scheduled prompts leave the reserved one out; a command samples it by index with the model's default steps
    assert [p["prompt"] for p in sampling.get_sample_prompts(args)] == ["a cat", "a dog"]
    forced = sampling.build_forced_prompts(args, indices=[1])
    assert forced[0]["prompt"] == "a reserved prompt" and "sample_steps" not in forced[0] and forced[0]["enum"] == 100
