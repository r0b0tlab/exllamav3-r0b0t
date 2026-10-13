from __future__ import annotations
from typing_extensions import override
import torch

from ..model.config import no_default
from ..modules import Linear, Embedding
from .dflash import DFlashConfig, DFlashModel

"""
Qwen3 DSpark drafter (openbmb/MiniCPM-V-4.7-35B-A3B-DSpark and the z-lab Qwen3.5 DSpark family):
a DFlash-style semi-autoregressive spec-decode draft —

  * backbone: 5 standard Qwen3 decoder layers (GQA, qk-norms) run as a query-only block
    forward over a context K/V cache precomputed from the target's tap hidden states
    (fc + hidden_norm over the concatenated taps, one k/v projection per layer);
  * markov head: low-rank sequential transition bias (markov_w1: V x r embedding of the
    previously sampled token, markov_w2: r x V projection) added to the base logits and
    sampled left-to-right — the "semi-autoregressive" part;
  * confidence head: per-position acceptance estimate sigmoid(w^T [h_k ; markov_w1[x_{k-1}]]),
    exported for dynamic draft-length scheduling;
  * embedding table and lm_head are shared with the target (not in the checkpoint).

Reference: vLLM qwen3_dspark.py / qwen3_dflash.py (Apache-2.0), DSpark paper arXiv:2607.05147.
"""


class Qwen3DSparkConfig(DFlashConfig):

    arch_string = "Qwen3DSparkModel"

    # Offset from the checkpoint's target_layer_ids to exl3 export indices. The original z-lab
    # DFlash release needed +1; variants whose reference reads hidden_states[id] directly use 0.
    # A/B against acceptance length if a new checkpoint family misbehaves; pin with "tap_shift".
    tap_shift = 1

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

        self.markov_rank = self.read_cfg(
            int, ["dflash_config->markov_rank", "markov_rank"], 256)
        self.draft_vocab_size = self.read_cfg(
            int, ["dflash_config->draft_vocab_size", "draft_vocab_size"], self.vocab_size)
        self.enable_confidence_head = self.read_cfg(
            bool, ["dflash_config->enable_confidence_head", "enable_confidence_head"], False)
        self.confidence_head_with_markov = self.read_cfg(
            bool, ["dflash_config->confidence_head_with_markov", "confidence_head_with_markov"], False)

        assert self.markov_rank > 0
        assert self.draft_vocab_size <= self.vocab_size or self.draft_vocab_size == self.vocab_size, \
            "reduced draft vocab with d2t mapping not supported"


class Qwen3DSparkModel(DFlashModel):
    config_class = Qwen3DSparkConfig

    def __init__(
        self,
        config: Qwen3DSparkConfig,
        **kwargs
    ):
        super().__init__(config, **kwargs)

        # Trunk boundary (deepseek_v4_mtp pattern): input layer + blocks + final norm.
        # Everything appended below is a side head addressed by sample_from_state
        # (markov_w1 takes token IDS, not the flowing state), so the sequential
        # forward must stop here -- see forward() below.
        self.fwd_end_idx = len(self.modules)

        r = config.markov_rank
        self.markov_w1 = Embedding(
            config = config,
            key = "markov_head.markov_w1",
            vocab_size = config.vocab_size,
            hidden_size = r,
        )
        # Device-resident (~75 MB at 150k vocab): the sequential markov sampling loop
        # then stays on-stream with GPU ids and no host syncs (as deepseek_v4_mtp)
        self.markov_w1.caps["prefer_cpu"] = False
        self.markov_w2 = Linear(
            config = config,
            key = "markov_head.markov_w2",
            in_features = r,
            out_features = config.draft_vocab_size,
        )
        self.modules += [self.markov_w1, self.markov_w2]

        self.confidence_head = None
        if config.enable_confidence_head:
            in_dim = config.hidden_size + (r if config.confidence_head_with_markov else 0)
            # fp32 weights, never quantized (out_features == 1 anyway)
            self.confidence_head = Linear(
                config = config,
                key = "confidence_head.proj",
                in_features = in_dim,
                out_features = 1,
                pad_to = 1,
            )
            self.modules += [self.confidence_head]

    def _markov_embed(self, ids_gpu: torch.Tensor) -> torch.Tensor:
        """(bsz,) ids -> (bsz, r) markov embedding, on the ids' device. Handles both the
        unquantized table (nn.Embedding, device ids) and the RowTable path (CPU ids).
        Output is cast to half: the table may load bf16, but markov_w2 is an fp16 GEMM."""
        import torch.nn as nn
        emb = self.markov_w1.embedding
        if isinstance(emb, nn.Embedding):
            return emb(ids_gpu).to(torch.half)
        return emb.forward(ids_gpu.cpu(), False).to(ids_gpu.device, torch.half)

    @override
    @torch.inference_mode
    def forward(self, input_ids: torch.Tensor, params: dict, **kwargs) -> torch.Tensor:
        """Draft-block forward: anchor id per row -> [anchor, noise x (block-1)] via the
        input layer (target embed + fc projection) -> blocks -> final norm. Returns the
        normed state (bsz, block, hidden) for sample_from_state. The markov/confidence
        heads are side modules outside the sequential chain (modules[:fwd_end_idx])."""
        params = params or {}
        x = self.prepare_inputs(input_ids, params)
        # Sampled-on-device seed ids for the markov loop; input_ids may be the
        # generator's pinned CPU staging buffer
        dev = self.modules[self.fwd_end_idx - 1].device
        params["dspark_seed_ids"] = input_ids.to(dev, non_blocking = True) if input_ids.device != dev else input_ids
        for m in self.modules[:self.fwd_end_idx]:
            x = m.prepare_for_device(x, params)
            x = m.forward(x, params)
        return x

    @override
    def sample_from_state(
        self,
        state: torch.Tensor,
        params: dict
    ) -> torch.Tensor:
        """Sequential markov sampling over the parallel backbone states.

        state: (1, block_size, hidden) fp16 output of the final norm. Row 0 is the anchor row;
        the generator crops it after sampling, exactly like DFlash. The markov bias for row k
        uses the id sampled at row k-1 (the anchor/seed id for row 0), reproducing the reference
        speculator's left-to-right pass. With the confidence head enabled, per-position
        acceptance estimates are exported as params["draft_confidence"] and the dynamic usable
        length as params["draft_confidence_len"] (positions before the first estimate below the
        0.5 threshold; the generator truncates the draft window to it).
        """
        target = self.attached_model()
        lm = target.modules[target.logit_layer_idx]
        logits = lm.forward(lm.prepare_for_device(state, params), params)
        logits = logits[..., : target.config.vocab_size].float()   # (1, block, V), fp32 as deepseek_v4_mtp

        bsz, block, _ = logits.shape
        device = logits.device

        # seed ids: the anchor token of this draft round. The draft forward's embedding step
        # stashes the block input ids (anchor in column 0) as params["input_ids"]
        seed_ids = params.get("dspark_seed_ids")
        if seed_ids is None:
            ids_in = params.get("input_ids")
            if ids_in is not None and ids_in.dim() == 2 and ids_in.shape[-1] == block:
                seed_ids = ids_in[:, 0].to(device)
            elif ids_in is not None and ids_in.dim() == 3 and ids_in.shape[-1] == block:
                seed_ids = ids_in[0, :, 0].to(device)
            else:
                seed_ids = torch.zeros(bsz, dtype = torch.long, device = device)
        else:
            seed_ids = seed_ids.view(-1).to(device)

        prev = seed_ids
        # Return contract matches deepseek_v4_mtp / DFlash: (bsz, block + 1) ids
        # [seed, p1..p_block], where p1 is the prediction at the anchor row. The
        # generator crops the seed column; returning bare predictions would shift
        # every draft one position late and drop the anchor prediction.
        out_ids = torch.empty(bsz, block + 1, dtype = torch.long, device = device)
        out_ids[:, 0] = seed_ids
        conf_vals = [torch.ones(bsz, device = device)] if self.confidence_head is not None else None

        with torch.inference_mode():
            for k in range(block):
                m_emb = self._markov_embed(prev)
                bias = self.markov_w2.forward(m_emb, params)
                # markov_w2's out_features may be padded past the vocab; logits are
                # already sliced to target vocab
                next_ids = torch.argmax(logits[:, k] + bias[..., : logits.shape[-1]].float(), dim = -1)
                out_ids[:, k + 1] = next_ids
                if conf_vals is not None:
                    if self.config.confidence_head_with_markov:
                        x = torch.cat([state[:, k], m_emb], dim = -1)
                    else:
                        x = state[:, k]
                    c = self.confidence_head.forward(x, params, out_dtype = torch.float).squeeze(-1)
                    conf_vals.append(torch.sigmoid(c))
                prev = next_ids

        if conf_vals is not None:
            conf = torch.stack(conf_vals, dim = -1)              # (bsz, block + 1), col 0 = anchor
            params["draft_confidence"] = conf
            # Truncation is opt-in: a miscalibrated sigmoid would otherwise gate drafting.
            # EXL3_DSPARK_CONF_LEN=1 caps the window at the first position below 0.5.
            import os as _os
            if _os.environ.get("EXL3_DSPARK_CONF_LEN", "0") != "0":
                below = conf < 0.5
                below[:, 0] = False                               # anchor position never gates
                first_bad = torch.argmax(below.to(torch.int8), dim = -1)
                none_bad = ~below.any(dim = -1)
                conf_len = torch.where(none_bad, torch.full_like(first_bad, block), first_bad)
                params["draft_confidence_len"] = int(conf_len.max().item())

        return out_ids
