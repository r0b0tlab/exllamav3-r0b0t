from __future__ import annotations
from typing_extensions import override

from .qwen3_5 import Qwen3_5VLMoeBaseConfig, Qwen3_5BaseModel

"""
MiniCPM-V 4.7 (openbmb/MiniCPM-V-4.7-35B-A3B), text side only.

The language model is architecturally a Qwen3.6-35B-A3B (transformers `qwen3_5_moe_text`: 30 gated-delta-net +
10 full-attention layers, 256 routed experts top-8 plus a gated shared expert, interleaved M-RoPE), so the whole
text stack is the stock Qwen3.5 MoE path with the checkpoint's `model.language_model` prefix. What differs:

- The vision tower (`model.vision_tower.*`: a SigLIP2-style NaViT with a window merger after block 6) and the
  projector (`model.merger.*`) are not modelled here. Their tensors are carried into the output pack unchanged
  (bf16) through get_additional_compiled_tensors, and `--vision_bits` is therefore inert. The Qwen3-VL vision
  config reader is bypassed on purpose: this checkpoint's preprocessor_config.json is not a Qwen2VL one.
- `mtp_num_hidden_layers` is 1 in the config but the checkpoint carries no `mtp.*` tensors, so no MTP model is
  registered and nothing is converted for it.
- vocab_size is 248144, not a multiple of 128. Linear zero-pads the lm_head output width to 248192; the logits are
  trimmed back to 248144 after the head so nothing downstream ever sees the pad columns.
"""


class MiniCPMV4_7Config(Qwen3_5VLMoeBaseConfig):
    arch_string = "MiniCPMV4_7ForConditionalGeneration"

    def __init__(
        self,
        directory: str,
        **kwargs,
    ):
        super().__init__(
            directory,
            "text_config",
            MiniCPMV4_7Model,
            None,       # vision model: tensors are passed through, not modelled
            None,       # mtp model: the checkpoint has no mtp tensors
            **kwargs,
        )


class MiniCPMV4_7Model(Qwen3_5BaseModel):
    config_class = MiniCPMV4_7Config

    def __init__(
        self,
        config: MiniCPMV4_7Config,
        **kwargs,
    ):
        super().__init__(
            config = config,
            key_prefix = "model.language_model",
            use_moe = True,
            **kwargs,
        )
        # Vocabulary is not a multiple of 128: keep the logits at the true width
        self.modules[self.logit_layer_idx].trim_padded_out = True

    @staticmethod
    @override
    def get_additional_compiled_tensors(config: MiniCPMV4_7Config) -> dict:
        return (
            config.stc.list_tensors(prefix = "model.vision_tower") |
            config.stc.list_tensors(prefix = "model.merger")
        )
