from __future__ import annotations
from typing_extensions import override
import torch

from .dflash import DFlashConfig, DFlashModel

"""
DSpark draft (Qwen3DSparkModel; openbmb/MiniCPM-V-4.7-35B-A3B-DSpark), quantization shim.

Attribution: the tensor layout, the head semantics and the depth convention below were read from vLLM
(vllm/model_executor/models/qwen3_dspark.py and vllm/v1/worker/gpu/spec_decode/dspark/speculator.py, Apache-2.0,
Copyright contributors to the vLLM project). No vLLM code is copied; this file only needs the names and shapes.

What the checkpoint holds (bf16, 62 tensors, 389,223,169 parameters, no embedding and no LM head: both are the target's):

    fc.weight                                  [2048, 10240]    EXL3   concatenated taps -> hidden
    hidden_norm.weight                         [2048]           verbatim (module walk misses it: extras fallback)
    layers.{0-4}.input_layernorm.weight        [2048]           module walk (norm)
    layers.{0-4}.post_attention_layernorm.w    [2048]           module walk (norm)
    layers.{0-4}.self_attn.q_proj / o_proj     [2048, 2048]     EXL3
    layers.{0-4}.self_attn.k_proj / v_proj     [512, 2048]      EXL3
    layers.{0-4}.self_attn.q_norm / k_norm     [256]            module walk (norm)
    layers.{0-4}.mlp.gate_proj / up_proj       [6144, 2048]     EXL3
    layers.{0-4}.mlp.down_proj                 [2048, 6144]     EXL3
    norm.weight                                [2048]           module walk (norm)
    markov_head.markov_w1.weight               [248144, 256]    verbatim (extras)
    markov_head.markov_w2.weight               [248144, 256]    verbatim (extras)
    confidence_head.proj.weight / .bias        [1, 2304] / [1]  verbatim (extras)

Every quantized linear has both dims divisible by 128, so nothing is padded (fc is built with pad_to = 1 anyway).
Converting this checkpoint as a plain DFlashDraftModel would silently DROP markov_head.* and confidence_head.*
(the compile step only carries what a module or get_additional_compiled_tensors names); this shim names them, and
the config reader below refuses a directory (source or pack) that lacks them.

How the open design questions were settled (from config.json and the vLLM source):

- sliding_window: null. The generic reader treats null as missing and DFlashConfig falls back to its 2048 default, but
  every layer_types entry is "full_attention" so Attention receives sliding_window = -1: the default is inert. The
  shim pins config.sliding_window = None and refuses anything but all-full-attention (vLLM refuses too).
- rope_parameters {rope_theta 1e7, rope_type "default", rope_dim 256}: the fork does not read rope_dim. Rotary width
  is head_dim * partial_rotary_factor = 256 = the whole head (vLLM's get_rope uses rope_dim as the rotary width, also
  256), NEOX style, plain 1-D positions. The shim asserts rope_dim is absent or equal to head_dim so a checkpoint that
  rotates a fraction of the head fails loudly instead of being run wrong. The target's M-RoPE does not apply to the
  draft.
- Taps: target_layer_ids [3, 11, 19, 27, 35] are the residual stream AFTER target layer N (vLLM adds 1 to its capture
  index): tap_shift = 0, the exllamav3 export index is used as is.
- Depth: block_size 15 is the draft's query-row count AND the number of drafted tokens (vLLM `sample_from_anchor`
  defaults to True: "anchor + N-1 noise, NOT 1 + N; every query position is a prediction", the anchor row predicts
  the first draft). config.json does not set sample_from_anchor, so the default applies.

NOT built here (marked TODO below):

- TODO(generator): the inherited DFlash generation path is NOT valid for this draft. DFlashModel advertises
  default_draft_size = block_size - 1 and the generator crops the anchor row (new_ids[:, 1:]) because a DFlash mask
  row predicts its own position. DSpark rows are shifted by one (anchor row = draft 1, 15 drafts per round) and
  the logits need the Markov bias (bias_k = markov_w1[x_{k-1}] @ markov_w2^T, sampled left to right; the engine
  walks the top-K candidates only). sample_from_state therefore raises: fork-side acceptance numbers from the stock
  generator would be misaligned. Drafting is left to a DSpark-capable inference engine.
- TODO(calibrated): a calibrated arm needs Hessians from real target taps. Approach: accumulate x^T x over ~1k target
  sequences for (a) the 5 concatenated taps (fc), (b) the hidden_norm output (k/v of all 5 layers) and (c) the block
  states (q/o, gate/up, down), then Linear.convert_exl3(H_data) as util/convert_mtp.py does. Until then the model
  is uncalibrated-only (caps["uncalibrated_quantize"], H = identity).
"""


class Qwen3DSparkConfig(DFlashConfig):

    arch_string = "Qwen3DSparkModel"

    # Taps are the output of target layer N, exactly the exllamav3 export index (vLLM captures at N + 1)
    tap_shift = 0

    # Tensor namespaces carried through compilation unchanged
    passthrough_prefixes = ("markov_head", "confidence_head")

    def __init__(
        self,
        directory: str,
        model_classes: dict | None = None,
        **kwargs,
    ):
        super().__init__(
            directory,
            model_classes or {"text": Qwen3DSparkModel},
            **kwargs
        )

        # Whole-context, non-causal full attention: no sliding window anywhere. sliding_window is null in the
        # checkpoint, the base reader substitutes 2048 for it, which stays inert only because of the check below
        assert all(t == "full_attention" for t in self.layer_types), \
            "Qwen3DSparkModel: every draft layer must be full_attention"
        assert not self.read_cfg(bool, "use_sliding_window", False), \
            "Qwen3DSparkModel: use_sliding_window is not supported"
        assert self.read_cfg(int, "sliding_window", None) is None, \
            "Qwen3DSparkModel: sliding_window is not supported"
        self.sliding_window = None

        # RoPE: the whole head is rotated with plain 1-D positions at rope_theta (default type)
        rope_dim = self.read_cfg(int, "rope_parameters->rope_dim", None)
        assert rope_dim is None or rope_dim == self.head_dim, \
            f"Qwen3DSparkModel: rope_dim {rope_dim} is not the head width {self.head_dim}, " \
            f"partial rotation is not implemented for this draft"
        assert self.read_cfg(str, "rope_parameters->rope_type", "default") == "default", \
            "Qwen3DSparkModel: only rope_type default is supported"
        assert self.read_cfg(float, "partial_rotary_factor", 1.0) == 1.0, \
            "Qwen3DSparkModel: partial_rotary_factor must be 1.0"

        # DSpark switches. Top-level keys are what vLLM reads, dflash_config repeats them: they must agree
        def both(expected_type, key, default):
            top = self.read_cfg(expected_type, key, None)
            nested = self.read_cfg(expected_type, f"dflash_config->{key}", None)
            assert top is None or nested is None or top == nested, \
                f"Qwen3DSparkModel: {key} is {top} at the top level and {nested} in dflash_config"
            v = top if top is not None else nested
            return default if v is None else v
        self.markov_rank = both(int, "markov_rank", None)
        assert self.markov_rank, "Qwen3DSparkModel: markov_rank is required"
        self.enable_confidence_head = both(bool, "enable_confidence_head", False)
        self.confidence_head_with_markov = both(bool, "confidence_head_with_markov", False)
        self.confidence_in = self.hidden_size + (self.markov_rank if self.confidence_head_with_markov else 0)
        self.assert_cfg(str, "dflash_config->markov_head_type", "vanilla", True)
        self.assert_cfg(str, "dflash_config->projector_type", "dspark", True)
        self.assert_cfg(str, "dflash_config->attention_mode", "gqa", True)
        self.assert_cfg(bool, "sample_from_anchor", True, True)
        self.draft_vocab_size = self.read_cfg(int, "draft_vocab_size", self.vocab_size)
        assert self.vocab_size is not None and self.draft_vocab_size == self.vocab_size, \
            "Qwen3DSparkModel: a reduced draft vocabulary (draft_vocab_size != vocab_size) is not supported"

        # The heads ride along as raw tensors: a directory without them would convert to an unusable draft
        self.check_head_tensors()


    def head_tensor_shapes(self) -> dict[str, list[int]]:
        shapes = {
            "markov_head.markov_w1.weight": [self.vocab_size, self.markov_rank],
            "markov_head.markov_w2.weight": [self.draft_vocab_size, self.markov_rank],
        }
        if self.enable_confidence_head:
            shapes["confidence_head.proj.weight"] = [1, self.confidence_in]
            shapes["confidence_head.proj.bias"] = [1]
        return shapes


    def check_head_tensors(self):
        found = {}
        for prefix in self.passthrough_prefixes:
            found.update(self.stc.list_tensors(prefix = prefix))
        want = self.head_tensor_shapes()
        for name, shape in want.items():
            assert name in found, f"Qwen3DSparkModel: {name} is missing from {self.directory}"
            assert list(found[name]["shape"]) == shape, \
                f"Qwen3DSparkModel: {name} has shape {found[name]['shape']}, expected {shape}"
        extra = sorted(set(found) - set(want))
        assert not extra, f"Qwen3DSparkModel: unexpected head tensors {extra}"


class Qwen3DSparkModel(DFlashModel):
    config_class = Qwen3DSparkConfig

    def __init__(
        self,
        config: Qwen3DSparkConfig,
        **kwargs
    ):
        super().__init__(config, **kwargs)
        self.caps.update({
            # H = identity quantization; the calibrated arm is a TODO (see the module docstring)
            "uncalibrated_quantize": True,
            # The anchor row is the first draft: block_size drafts per round (DFlash: block_size - 1)
            "default_draft_size": config.block_size,
        })


    @override
    def attach_to(self, target):
        if target.loaded_tp:
            raise NotImplementedError("Qwen3DSparkModel does not support tensor-parallel targets")
        if target.config.vocab_size != self.config.vocab_size:
            raise ValueError(
                f"DSpark vocabulary size {self.config.vocab_size} does not match "
                f"target vocabulary size {target.config.vocab_size}"
            )
        if target.config.hidden_size != self.config.hidden_size:
            raise ValueError(
                f"DSpark hidden size {self.config.hidden_size} does not match "
                f"target hidden size {target.config.hidden_size}"
            )
        if not 0 <= self.config.mask_token_id < target.config.vocab_size:
            raise ValueError("DSpark mask_token_id is outside the target vocabulary")
        if any(not 0 <= layer_id < target.config.num_hidden_layers
               for layer_id in self.config.target_layer_ids):
            raise ValueError("DSpark target_layer_ids contains a layer outside the target model")
        super().attach_to(target)


    @override
    def sample_from_state(
        self,
        state: torch.Tensor,
        params: dict
    ) -> torch.Tensor:
        raise NotImplementedError(
            "Qwen3DSparkModel: the stock DFlash sampling and anchor-crop path is not valid for DSpark "
            "(anchor row is draft 1, the Markov head biases the logits). The fork converts and loads this "
            "draft; drafting is left to a DSpark-capable inference engine. See the module docstring, TODO(generator)."
        )


    @classmethod
    @override
    def get_additional_compiled_tensors(cls, config: Qwen3DSparkConfig) -> dict:
        # hidden_norm (stored in DFlashInputLayer under a key outside the fc prefix) plus the Markov tables and the
        # confidence head: raw tensors outside any Linear, copied bit for bit from the source
        tensors = dict(super().get_additional_compiled_tensors(config))
        for prefix in config.passthrough_prefixes:
            tensors.update(config.stc.list_tensors(prefix = prefix))
        return tensors
