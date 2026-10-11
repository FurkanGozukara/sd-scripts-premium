"""Compatibility helpers for running sd-scripts with transformers 5.

transformers 5 flattened ``CLIPTextModel``: the former inner ``CLIPTextTransformer`` became the model itself, so the
model no longer has a ``text_model`` sub-module and its state-dict keys lost the ``text_model.`` prefix.
sd-scripts relies on the transformers 4 layout in several places:

* checkpoints are loaded with ``load_state_dict`` using ``text_model.*`` keys (FLUX CLIP-L, SDXL text encoder 1, SD1/2),
* training code accesses ``text_encoder.text_model.*`` (e.g. freezing the last layer of SDXL text encoder 1),
* LoRA module names are derived from the module path (``lora_te1_text_model_encoder_layers_...``) and saved
  SDXL checkpoints use ``conditioner.embedders.0.transformer.text_model.*`` keys.

``CLIPTextModelWithProjection`` kept its ``text_model`` child in transformers 5, so only ``CLIPTextModel`` needs the
wrapper below. It follows the same pattern as transformers' own ``CLIPTextModelWithProjection`` (a
``CLIPPreTrainedModel`` whose ``text_model`` is the flattened model), which restores the transformers 4 module tree,
state-dict keys and outputs.
"""

from functools import lru_cache

import torch
from transformers import CLIPTextConfig, CLIPTextModel
from transformers.models.clip.modeling_clip import CLIPPreTrainedModel


@lru_cache(maxsize=1)
def clip_text_model_is_flat() -> bool:
    """True when ``transformers.CLIPTextModel`` has no ``text_model`` sub-module (transformers >= 5)."""
    config = CLIPTextConfig(
        vocab_size=8,
        hidden_size=8,
        intermediate_size=8,
        num_hidden_layers=1,
        num_attention_heads=1,
        max_position_embeddings=4,
        pad_token_id=1,
        bos_token_id=0,
        eos_token_id=2,
    )
    with torch.device("meta"):
        model = CLIPTextModel(config)
    return not hasattr(model, "text_model")


class CLIPTextModelV4Layout(CLIPPreTrainedModel):
    """``CLIPTextModel`` with the transformers 4 module layout (``self.text_model``) on top of transformers 5."""

    config: CLIPTextConfig
    input_modalities = ("text",)

    def __init__(self, config):
        super().__init__(config)
        self.text_model = CLIPTextModel._from_config(config)
        self.post_init()

    def get_input_embeddings(self) -> torch.nn.Module:
        return self.text_model.embeddings.token_embedding

    def set_input_embeddings(self, value):
        self.text_model.embeddings.token_embedding = value

    def forward(self, *args, **kwargs):
        return self.text_model(*args, **kwargs)


def clip_text_model_from_config(config) -> torch.nn.Module:
    """Build a ``CLIPTextModel`` with the transformers 4 layout (``text_model.*`` keys) on any transformers version."""
    if clip_text_model_is_flat():
        return CLIPTextModelV4Layout._from_config(config)
    return CLIPTextModel._from_config(config)
