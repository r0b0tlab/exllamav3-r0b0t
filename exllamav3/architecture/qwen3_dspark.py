from __future__ import annotations
from typing_extensions import override
import torch

from ..modules.arch_specific.dspark_markov import (
    DSparkHeads, markov_candidates, markov_walk, markov_confidence, adaptive_window,
)
from ..util.device_copy import to_device
from ..util.tensor import get_for_device
from .dflash import DFlashConfig, DFlashModel

"""
DSpark draft (Qwen3DSparkModel; openbmb/MiniCPM-V-4.7-35B-A3B-DSpark): config reader, quantization shim, drafting.

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

Generation (Qwen3DSparkModel.sample_from_state):

- One draft round is one forward over the block [anchor, mask x (block_size - 1)] at positions P .. P + block_size - 1
  (P = number of target-processed tokens), attending non-causally to the committed context rows and the whole block.
  Every block row is a next-token prediction (the anchor row predicts the first draft), so a round yields block_size
  drafts, and the generator verifies block_size + 1 target rows (anchor + drafts). The sampler returns the ids in the
  layout [anchor, drafts...] the generator's DFlash path already consumes (it crops the leading anchor).
- The base logits come from the target's LM head. The Markov bias markov_w1[x_{k-1}] @ markov_w2^T is evaluated for the
  top-K base candidates of each row only (K = markov_top_k, default 64; the reference measured -3.8% accepted length
  against the full-vocabulary walk at K = 64, -6% at 32, -9% at 16) by the DFlash2 selector-walk kernels.
- The proposals are greedy, so verification by sample-and-compare is exact for greedy and for sampled decoding.
- The confidence head is exported through params["draft_conf"] (dynamic_draft_tokens) and, when draft_conf_threshold
  or adaptive_verification is set, as params["draft_confidence_len"] (the number of drafts worth verifying: leading
  confidences above the threshold, or the count that maximizes expected tokens per unit of time under a cost model
  of a round); neither changes the output, only how many drafts are verified.

TODO(calibrated): a calibrated arm needs Hessians from real target taps. Approach: accumulate x^T x over ~1k target
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

    # Directory of the Markov / confidence tensors (module key of DSparkHeads)
    key_heads = "markov_head"

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

        # Raw Markov / confidence tensors, identity in the module walk (after the final norm)
        self.heads = DSparkHeads(
            config = config,
            key = self.key_heads,
            vocab_size = config.vocab_size,
            rank = config.markov_rank,
            confidence_in = config.confidence_in if config.enable_confidence_head else None,
        )
        self.modules += [self.heads]

        # Candidates per row the Markov bias is evaluated for (top-K of the base logits)
        self._markov_top_k = 64
        # With the confidence head: keep only the leading drafts whose confidence is at least this (None: all)
        self.draft_conf_threshold = None
        # With the confidence head: verify the number of drafts that maximizes the expected accepted tokens per unit
        # of time under a cost model of the round (see set_verify_cost)
        self.adaptive_verification = False
        self.verify_cost = [1.0 + 0.1 * n for n in range(config.block_size + 1)]
        self.draft_cost = 0.35


    def set_verify_cost(self, verify_cost: list[float], draft_cost: float = 0.0):
        """
        Cost model for adaptive_verification, in any one unit (e.g. milliseconds).

        :param verify_cost:
            Cost of one target verify pass over 1 .. block_size + 1 rows (anchor + n drafts), entry n. The default
            is a plain decode step of 1 plus 0.1 per extra row, typical for a mixture-of-experts target on a
            bandwidth-bound device. Measure it for a different target or device.

        :param draft_cost:
            Cost of producing the drafts and updating the draft's context, same unit.
        """
        assert len(verify_cost) >= self.config.block_size + 1, \
            "verify_cost has one entry per verified row count, 1 .. block_size + 1"
        self.verify_cost = [float(c) for c in verify_cost]
        self.draft_cost = float(draft_cost)


    @property
    def markov_top_k(self) -> int:
        return self._markov_top_k


    @markov_top_k.setter
    def markov_top_k(self, k: int):
        if not 1 <= k <= self.config.vocab_size:
            raise ValueError(f"markov_top_k must be between 1 and the vocabulary size, got {k}")
        self._markov_top_k = int(k)


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
        if target.logit_layer_idx is None:
            raise ValueError("DSpark target has no compatible LM head")
        super().attach_to(target)


    @override
    def prepare_inputs(self, input_ids: torch.Tensor, params: dict) -> torch.Tensor:
        assert input_ids.shape[-1] == 1, \
            "DSpark expects one verified anchor token per draft block"
        # The Markov walk starts from the anchor token
        params["dspark_anchor_ids"] = input_ids
        return super().prepare_inputs(input_ids, params)


    @override
    def sample_from_state(
        self,
        state: torch.Tensor,
        params: dict
    ) -> torch.Tensor:
        """
        Target LM head over all block rows, top-K candidates per row from the base logits, then the Markov walk over
        the candidates, left to right from the anchor. Row k predicts the token after row k, so all block_size rows
        are drafts. Returns (bsz, block_size + 1) ids [anchor, drafts...]: the generator crops the anchor.

        With params["export_draft_conf"], params["draft_conf"] receives (bsz, block_size + 1) scores [0, confidence...]
        aligned with those ids. With draft_conf_threshold set, params["draft_confidence_len"] receives the longest
        prefix of drafts, over the batch, whose confidence is at least the threshold (0: skip drafting this round)
        """
        target = self.attached_model()
        lm = target.modules[target.logit_layer_idx]
        logits = lm.prepare_for_device(state.half(), params)
        logits = lm.forward(logits, params)

        heads = self.heads
        dev = heads.device
        logits = to_device(logits, dev)
        # The generator stages the anchor ids in pinned memory: upload without a host sync
        anchor = get_for_device(params, "dspark_anchor_ids", dev)[:, -1]

        vocab = target.config.vocab_size
        top_k = min(self._markov_top_k, vocab)
        unary, cands = markov_candidates(logits, vocab, top_k)
        out = markov_walk(unary, cands, anchor, heads.w1, heads.w2)

        thr = self.draft_conf_threshold
        adaptive = self.adaptive_verification and heads.has_confidence
        if params.get("export_draft_conf") or thr is not None or adaptive:
            bsz, rows = out.shape[0], out.shape[1] - 1
            if heads.has_confidence:
                conf = markov_confidence(to_device(state, dev), out[:, :-1], heads.w1, heads.conf_w, heads.conf_b)
            else:
                # No confidence head: the winning base logit separates good from degenerate rows
                conf = unary[..., 0]
            if params.get("export_draft_conf"):
                params["draft_conf"] = torch.cat((conf.new_zeros((bsz, 1)), conf), dim = 1)
            keep = None
            if thr is not None and heads.has_confidence:
                run = torch.cumprod((conf >= thr).to(torch.int32), dim = 1).sum(dim = 1)
                keep = int(run.max().item())
            if adaptive:
                n = adaptive_window(conf, self.verify_cost, self.draft_cost)
                keep = n if keep is None else min(keep, n)
            if keep is not None:
                params["draft_confidence_len"] = keep
        return out


    @classmethod
    @override
    def get_additional_compiled_tensors(cls, config: Qwen3DSparkConfig) -> dict:
        # hidden_norm (stored in DFlashInputLayer under a key outside the fc prefix) plus the Markov tables and the
        # confidence head: raw tensors outside any Linear, copied bit for bit from the source
        tensors = dict(super().get_additional_compiled_tensors(config))
        for prefix in config.passthrough_prefixes:
            tensors.update(config.stc.list_tensors(prefix = prefix))
        return tensors
