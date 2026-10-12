# Input-axis DoRA for sd-scripts' LoRA modules (FLUX.1 and SDXL, U-Net and text encoders):
#     W' = (W + s*B@A) * dora_scale / ||W + s*B@A||_col
# "<module>.dora_scale" of shape (1, in) [Conv2d: (1, in, 1, 1)] is the LyCORIS / kohya layout that ComfyUI, SwarmUI,
# Forge and A1111 apply exactly. The math is musubi-tuner's (networks/lora_shared.py, installed in the same
# environment), so both trainers train the same DoRA. Enable it with the network argument use_dora=True.

import torch


def _shared():
    try:
        from musubi_tuner.networks import lora_shared
    except ImportError as exc:  # sd-scripts used on its own
        raise ImportError("DoRA needs musubi-tuner in this Python environment (pip install -e musubi-tuner)") from exc
    return lora_shared


def is_true(value) -> bool:
    return str(value).strip().strip("'\"").casefold() in ("1", "true", "yes", "on")


def has_dora_weights(weights_sd) -> bool:
    return any(key.endswith(".dora_scale") for key in weights_sd.keys())


_CLASSES = {}


def dora_class(base_class):
    """The input-axis DoRA variant of an sd-scripts LoRAModule class."""
    if base_class in _CLASSES:
        return _CLASSES[base_class]
    shared = _shared()

    class DoRAModule(base_class):
        def __init__(self, lora_name, org_module, multiplier=1.0, lora_dim=4, alpha=1, *args, **kwargs):
            super().__init__(lora_name, org_module, multiplier, lora_dim, alpha, *args, **kwargs)
            if getattr(self, "ggpo_sigma", None) is not None and getattr(self, "ggpo_beta", None) is not None:
                raise ValueError("DoRA cannot be combined with LoRA-GGPO (ggpo_sigma / ggpo_beta)")
            self.dora_base = shared.DoRABaseWeight(org_module)
            is_conv = org_module.__class__.__name__ in ("Conv2d", "Conv3d")
            kernel_size = tuple(org_module.kernel_size) if is_conv else ()
            self._dora_channel_dim = 1 if is_conv else -1
            with torch.no_grad():
                weight = org_module.weight  # a weight still on the CPU is read through the GPU (much faster)
                device = None
                if not weight.is_cuda and weight.device.type != "meta" and torch.cuda.is_available():
                    device = torch.device("cuda", torch.cuda.current_device())
                self._dora_col_sq = shared.dora_column_norm_sq_of(self.dora_base, device=device)
                magnitude = self._dora_col_sq.view(-1, self.dora_base.kernel).sum(dim=1).sqrt()
            downs, _ = self._dora_weights()
            magnitude = magnitude.reshape(1, -1, *([1] * len(kernel_size))).to(device=downs[0].device, dtype=downs[0].dtype)
            self.dora_scale = torch.nn.Parameter(magnitude)
            self._dora_cache = None
            self._dora_from_lora_prefix = None
            self.register_load_state_dict_post_hook(DoRAModule._dora_after_load)

        def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
            # a plain LoRA continued as DoRA gets magnitudes ||W + s*B@A||_col, so its first step reproduces the LoRA
            # (set in the post hook: lora_down / lora_up load after this module's own tensors)
            if f"{prefix}dora_scale" not in state_dict and any(key.startswith(f"{prefix}lora_up") for key in state_dict):
                self._dora_from_lora_prefix = prefix
            super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

        @staticmethod
        def _dora_after_load(module, incompatible_keys):
            prefix = getattr(module, "_dora_from_lora_prefix", None)
            if prefix is None:
                return
            module._dora_from_lora_prefix = None
            with torch.no_grad():
                downs, ups = module._dora_weights()
                norm = shared.dora_column_norm(module.dora_base, downs, ups, float(module.multiplier) * float(module.scale), module._dora_col_sq)
                module.dora_scale.copy_(norm.reshape(module.dora_scale.shape).to(module.dora_scale.device, module.dora_scale.dtype))
            module._dora_cache = None
            if f"{prefix}dora_scale" in incompatible_keys.missing_keys:
                incompatible_keys.missing_keys.remove(f"{prefix}dora_scale")

        def _dora_weights(self):
            if isinstance(self.lora_down, torch.nn.ModuleList):
                return [module.weight for module in self.lora_down], [module.weight for module in self.lora_up]
            return [self.lora_down.weight], [self.lora_up.weight]

        def dora_factor(self, device) -> torch.Tensor:
            """dora_scale / (||W + s*B@A||_col + eps) for every input channel (fp32); reused while nothing changes in
            forwards without gradients (sampling)."""
            # nothing is stored while torch.compile traces: Dynamo stops compiling a block whose activation-
            # checkpointed forward sets attributes
            compiling = torch.compiler.is_compiling()
            key = None
            if not torch.is_grad_enabled() and not compiling:
                key = (device, float(self.multiplier), tuple((p.data_ptr(), p._version) for p in self.parameters()))
                if self._dora_cache is not None and self._dora_cache[0] == key:
                    return self._dora_cache[1]
            col_sq = self._dora_col_sq
            if col_sq.device != device:
                col_sq = col_sq.to(device)
                if not compiling:
                    self._dora_col_sq = col_sq
            downs, ups = self._dora_weights()
            factor = shared.dora_input_factor(self.dora_base, downs, ups, float(self.multiplier) * float(self.scale), col_sq, self.dora_scale)
            if key is not None:
                self._dora_cache = (key, factor)
            return factor

        def _down(self, module, x, factor):
            if isinstance(module, torch.nn.Linear):
                return shared.scaled_down_projection(x, module.weight, factor)
            weight = module.weight
            weight = (weight.float() * factor.view(1, -1, *([1] * (weight.dim() - 2)))).to(weight.dtype)
            return module._conv_forward(x, weight, None)

        def forward(self, x):
            if not getattr(self, "enabled", True):
                return self.org_forward(x)
            if self.module_dropout is not None and self.training and torch.rand(1) < self.module_dropout:
                return self.org_forward(x)
            factor = self.dora_factor(x.device)
            # the frozen base layer gets the scaled input; the LoRA path keeps x (shared by q / k / v, kept once for
            # backward as with LoRA) and scales A's columns instead: B (A f) x = B A (f x)
            org_forwarded = self.org_forward(shared.scale_input_channels(x, factor, self._dora_channel_dim))
            split = isinstance(self.lora_down, torch.nn.ModuleList)
            downs = list(self.lora_down) if split else [self.lora_down]
            ups = list(self.lora_up) if split else [self.lora_up]
            lxs = [self._down(module, x, factor) for module in downs]
            if self.dropout is not None and self.training:
                lxs = [torch.nn.functional.dropout(lx, p=self.dropout) for lx in lxs]
            scale = self.scale
            if self.rank_dropout is not None and self.training:
                masked = []
                for lx in lxs:
                    mask = torch.rand((lx.size(0), self.lora_dim), device=lx.device) > self.rank_dropout
                    if lx.dim() == 3:
                        mask = mask.unsqueeze(1)
                    elif lx.dim() == 4:
                        mask = mask.unsqueeze(-1).unsqueeze(-1)
                    masked.append(lx * mask)
                lxs = masked
                scale = self.scale * (1.0 / (1.0 - self.rank_dropout))
            lxs = [up(lx) for up, lx in zip(ups, lxs)]
            delta = torch.cat(lxs, dim=-1) if split else lxs[0]
            return org_forwarded + delta * self.multiplier * scale

    DoRAModule.__name__ = DoRAModule.__qualname__ = f"DoRA{base_class.__name__}"
    _CLASSES[base_class] = DoRAModule
    return DoRAModule


def saved_dtype(key: str, value: torch.Tensor, dtype):
    """A DoRA scale multiplies the whole base weight: it is saved in fp32 (bf16 rounds each column by up to 0.4%)."""
    if key.endswith(".dora_scale") and value.is_floating_point():
        return torch.float32
    return dtype
