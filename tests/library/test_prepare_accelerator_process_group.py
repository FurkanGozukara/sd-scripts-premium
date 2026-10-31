import argparse
from unittest.mock import patch

from accelerate import InitProcessGroupKwargs

from library import train_util


def _args():
    return argparse.Namespace(
        logging_dir=None,
        log_prefix=None,
        log_with=None,
        wandb_api_key=None,
        torch_compile=False,
        dynamo_backend="inductor",
        ddp_timeout=None,
        ddp_gradient_as_bucket_view=False,
        ddp_static_graph=False,
        gradient_accumulation_steps=1,
        mixed_precision="no",
        deepspeed=False,
    )


def _kwargs_handlers(monkeypatch, local_rank, device_count=2):
    if local_rank is None:
        monkeypatch.delenv("LOCAL_RANK", raising=False)
    else:
        monkeypatch.setenv("LOCAL_RANK", str(local_rank))
    with (
        patch.object(train_util.torch.cuda, "device_count", return_value=device_count),
        patch.object(train_util, "Accelerator") as accelerator,
    ):
        train_util.prepare_accelerator(_args())
    return accelerator.call_args.kwargs["kwargs_handlers"]


def test_single_process_on_a_multi_gpu_machine_gets_no_process_group_kwargs(monkeypatch):
    # plain `python train.py` (GPU IDs left empty) on a 2-GPU PC: an explicit backend made accelerate 1.15 look for a
    # process group that was never created ("Default process group has not been initialized")
    handlers = _kwargs_handlers(monkeypatch, local_rank=None)

    assert not any(isinstance(h, InitProcessGroupKwargs) for h in handlers)


def test_distributed_launch_keeps_the_process_group_kwargs(monkeypatch):
    handlers = _kwargs_handlers(monkeypatch, local_rank=0)

    assert any(isinstance(h, InitProcessGroupKwargs) for h in handlers)


def test_single_gpu_machine_gets_no_process_group_kwargs(monkeypatch):
    handlers = _kwargs_handlers(monkeypatch, local_rank=0, device_count=1)

    assert not any(isinstance(h, InitProcessGroupKwargs) for h in handlers)
