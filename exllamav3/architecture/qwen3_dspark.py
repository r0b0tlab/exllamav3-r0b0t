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
        self._sts_temps = None     # STS per-position exponents, loaded from EXL3_DSPARK_STS
        self._sts_sps = None       # verify-batch -> round-ms table, loaded from EXL3_DSPARK_SPS
        self._cap = None           # CUDA graph bundle for the draft+sample round
        self._cap_failed = False
        self._stash = None         # (state, out_ids, conf, params) from the last forward
        self._rounds_seen = 0
        self._pin = None           # pinned staging for the embedded draft block
        self._pin_dev = None       # stable device tensor the trunk reads
        self._pin_shape = None
        self._pin_n = 0
        self._state_stage = None   # stable device copy of the trunk state for the graph
        self._seed_stage = None    # stable device copy of the anchor id for the graph
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

    def _markov_sample(self, state: torch.Tensor, seed_ids: torch.Tensor, params: dict):
        """Sequential markov sampling over the parallel backbone states. Pure tensor ops,
        CUDA-graph-capturable. Returns (out_ids (bsz, block+1), conf or None)."""
        target = self.attached_model()
        lm = target.modules[target.logit_layer_idx]
        logits = lm.forward(lm.prepare_for_device(state, params), params)
        logits = logits[..., : target.config.vocab_size].float()   # (1, block, V), fp32 as deepseek_v4_mtp

        bsz, block, _ = logits.shape
        device = logits.device
        seed = seed_ids.view(-1).to(device)

        prev = seed
        # Return contract matches deepseek_v4_mtp / DFlash: (bsz, block + 1) ids
        # [seed, p1..p_block], where p1 is the prediction at the anchor row. The
        # generator crops the seed column; returning bare predictions would shift
        # every draft one position late and drop the anchor prediction.
        out_ids = torch.empty(bsz, block + 1, dtype = torch.long, device = device)
        out_ids[:, 0] = seed
        conf_vals = [torch.ones(bsz, device = device)] if self.confidence_head is not None else None

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

        conf = torch.stack(conf_vals, dim = -1) if conf_vals is not None else None
        return out_ids, conf

    def _stage_trunk_input(self, x: torch.Tensor):
        """Copy the input layer's output (possibly unpinned CPU, if the target's token
        embedding is CPU-resident) into stable staging: host -> pinned -> stable device
        tensor. The eager prologue refreshes it every round, so a captured graph can read
        the device tensor directly."""
        n = x.numel()
        if self._pin_dev is None or self._pin_dev.numel() < n:
            self._pin = torch.empty(n, dtype = x.dtype, pin_memory = True)
            self._pin_dev = torch.empty(n, dtype = x.dtype, device = "cuda")
        self._pin_shape = tuple(x.shape)
        self._pin_n = n
        self._pin[:n].copy_(x.reshape(-1))
        self._pin_dev[:n].copy_(self._pin[:n], non_blocking = True)
        return self._pin_dev[:n].view(x.shape)

    def _prologue(self, input_ids: torch.Tensor, params: dict):
        """Eager per-round part: attention prep + input layer (CPU embedding possible) +
        staging refresh. Runs before both the eager body and a graph replay."""
        x = self.prepare_inputs(input_ids, params)
        m0 = self.modules[0]
        x = m0.prepare_for_device(x, params)
        x = m0.forward(x, params)                       # (bsz, block, hidden), maybe CPU
        self._stage_trunk_input(x)

    def _body(self, input_ids: torch.Tensor, params: dict):
        """GPU-only trunk: staged embedded block -> blocks -> final norm -> state.
        The markov sampling runs separately (eager or graph-replayed)."""
        x = self._pin_dev[:self._pin_n].view(self._pin_shape)
        for m in self.modules[1:self.fwd_end_idx]:
            x = m.prepare_for_device(x, params)
            x = m.forward(x, params)
        return x

    def _trunk(self, input_ids: torch.Tensor, params: dict) -> torch.Tensor:
        """Eager trunk: input layer + blocks + final norm -> (bsz, block, hidden)."""
        self._prologue(input_ids, params)
        return self._body(input_ids, params)

    def _graph_sig(self, input_ids: torch.Tensor, params: dict):
        """Capture identity: the input tensor plus every tensor param the captured kernels
        read. A change (staging-buffer growth, different cache) means the graph's baked
        pointers are stale."""
        sig = [input_ids.data_ptr(), tuple(input_ids.shape), params.get("attn_mode")]
        for k in ("block_table", "cache_seqlens", "cache"):
            v = params.get(k)
            sig.append((k, v.data_ptr() if torch.is_tensor(v) else id(v)))
        return tuple(sig)

    def _markov_body(self, params: dict):
        """Markov sampling from the staged state (self._state_stage / self._seed_stage),
        GPU-only, CUDA-graph-capturable. The trunk (attention kernels reject capture:
        exllamav3_ext graph.cu guard) stays eager."""
        state = self._state_stage
        seed = self._seed_stage
        target = self.attached_model()
        lm = target.modules[target.logit_layer_idx]
        logits = lm.forward(lm.prepare_for_device(state, params), params)
        logits = logits[..., : target.config.vocab_size].float()

        bsz, block, _ = logits.shape
        device = logits.device

        prev = seed.view(-1).to(device)
        out_ids = torch.empty(bsz, block + 1, dtype = torch.long, device = device)
        out_ids[:, 0] = prev
        conf_vals = [torch.ones(bsz, device = device)] if self.confidence_head is not None else None

        for k in range(block):
            m_emb = self._markov_embed(prev)
            bias = self.markov_w2.forward(m_emb, params)
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

        conf = torch.stack(conf_vals, dim = -1) if conf_vals is not None else None
        return out_ids, conf

    def _try_capture(self, input_ids: torch.Tensor, params: dict, sig):
        """Capture the markov sampling loop only (~4 ms of small-kernel launch overhead per
        round). The draft trunk cannot be captured: the attention extension rejects stream
        capture."""
        import os as _os
        try:
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                self._markov_body(params)             # warmup on side stream
            torch.cuda.current_stream().wait_stream(s)
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                out_ids, conf = self._markov_body(params)
            self._cap = {"graph": g, "sig": sig, "out_ids": out_ids, "conf": conf}
        except Exception as e:
            import traceback as _tb
            if _os.environ.get("EXL3_DSPARK_GRAPH_DEBUG"):
                _tb.print_exc()
            print(f" -- dspark markov graph capture failed ({type(e).__name__}: {e}); staying eager")
            self._cap_failed = True

    @override
    @torch.inference_mode
    def forward(self, input_ids: torch.Tensor, params: dict, **kwargs) -> torch.Tensor:
        """Draft-block forward (trunk + markov sampling + confidence). Returns the normed
        state (bsz, block, hidden); the sampled ids/confidence are stashed for
        sample_from_state. The markov sampling loop is CUDA-graph-replayed when
        EXL3_DSPARK_GRAPH != 0 (default on): it is ~4 ms of small-kernel launch overhead
        per round, which a graph reduces to well under 1 ms. The trunk stays eager (the
        attention extension rejects stream capture)."""
        params = params or {}
        self._rounds_seen += 1
        import os as _os
        use_graph = (_os.environ.get("EXL3_DSPARK_GRAPH", "0") != "0"   # opt-in: no gain on
                     # 3090 (markov loop is bandwidth-bound: 15 re-reads of the 127 MB
                     # markov_w2 table), but correct and may help launch-bound configs
                     and not self._cap_failed
                     and input_ids.dim() == 2 and input_ids.shape[-1] == 1
                     and "block_table" in params)
        sig = None
        if use_graph:
            sig = self._graph_sig(input_ids, params)
            cap = self._cap
            if cap is not None:
                if cap["sig"] == sig:
                    state = self._trunk(input_ids, params)
                    self._stage_for_graph(state, input_ids)
                    cap["graph"].replay()
                    self._stash = (state, cap["out_ids"], cap["conf"], params)
                    return state
                self._cap = None      # stale pointers (staging growth); recapture below
        state = self._trunk(input_ids, params)
        dev = state.device
        seed = input_ids.to(dev, non_blocking = True) if input_ids.device != dev else input_ids
        out_ids, conf = self._markov_sample(state, seed, params)
        if use_graph and self._cap is None and self._rounds_seen >= 3:
            self._stage_for_graph(state, input_ids)
            self._try_capture(input_ids, params, sig)
        self._stash = (state, out_ids, conf, params)
        return state

    def _stage_for_graph(self, state: torch.Tensor, input_ids: torch.Tensor):
        """Refresh the stable device tensors the captured markov body reads."""
        dev = state.device
        if self._state_stage is None or self._state_stage.shape != state.shape:
            self._state_stage = torch.empty_like(state)
            self._seed_stage = torch.empty(input_ids.shape, dtype = input_ids.dtype, device = dev)
        self._state_stage.copy_(state)
        self._seed_stage.copy_(input_ids.to(dev, non_blocking = True) if input_ids.device != dev else input_ids, non_blocking = True)

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
        length as params["draft_confidence_len"].
        """
        stash = self._stash
        if stash is not None and stash[0] is state:
            out_ids, conf = stash[1], stash[2]
        else:
            # Direct callers (or a foreign state): resolve the seed and sample eagerly
            seed_ids = params.get("dspark_seed_ids")
            if seed_ids is None:
                ids_in = params.get("input_ids")
                block = state.shape[-2]
                if ids_in is not None and ids_in.dim() == 2 and ids_in.shape[-1] == block:
                    seed_ids = ids_in[:, 0]
                elif ids_in is not None and ids_in.dim() == 3 and ids_in.shape[-1] == block:
                    seed_ids = ids_in[0, :, 0]
                else:
                    seed_ids = torch.zeros(state.shape[0], dtype = torch.long, device = state.device)
            out_ids, conf = self._markov_sample(state, seed_ids, params)
        block = out_ids.shape[-1] - 1

        if conf is not None:
            params["draft_confidence"] = conf
            params["draft_conf"] = conf     # key read by the generator's DraftConfidenceCalibrator
            import os as _os, json as _json
            _sink = _os.environ.get("EXL3_DSPARK_CONF_DUMP")
            if _sink:
                with open(_sink, "a") as _f:
                    _f.write(_json.dumps(conf[0].tolist()) + "\n")

            # Theta-scheduled verification length (DSpark arXiv:2607.05147 Alg. 1, single
            # request): STS-calibrated prefix survival a_j = prod_i c_i^T_i, admit l while
            # Theta_l = (1 + sum_j<=l a_j) / T_round(l+1) increases; stop at the first
            # decline. Early stopping keeps the scheduler non-anticipating => lossless
            # (paper Appendix A). Opt-in via EXL3_DSPARK_STS=<temps.json> (+EXL3_DSPARK_SPS=
            # <sps.json> mapping verify-batch size -> round ms); l = 0 means skip drafting.
            _sts = _os.environ.get("EXL3_DSPARK_STS")
            if _sts:
                if self._sts_temps is None:
                    self._sts_temps = _json.load(open(_sts))
                    _sps_path = _os.environ.get("EXL3_DSPARK_SPS")
                    assert _sps_path, "EXL3_DSPARK_STS requires EXL3_DSPARK_SPS=<json>"
                    self._sts_sps = {int(k): float(v) for k, v in _json.load(open(_sps_path)).items()}
                params["draft_confidence_len"] = self._theta_window(conf)

            # Fallback heuristic, unchanged: EXL3_DSPARK_CONF_LEN=1 caps the window at the
            # first position below 0.5.
            elif _os.environ.get("EXL3_DSPARK_CONF_LEN", "0") != "0":
                below = conf < 0.5
                below[:, 0] = False                               # anchor position never gates
                first_bad = torch.argmax(below.to(torch.int8), dim = -1)
                none_bad = ~below.any(dim = -1)
                conf_len = torch.where(none_bad, torch.full_like(first_bad, block), first_bad)
                params["draft_confidence_len"] = int(conf_len.max().item())

        return out_ids

    def _theta_window(self, conf: torch.Tensor) -> int:
        """Admission count l maximizing expected tokens/second, batch row 0 (bsz=1 eval).

        Global argmax over l in [0, block] rather than the paper's early-stopping greedy:
        this engine's round-time curve is non-unimodal (a ~17 ms fixed per-round floor for
        the draft machinery makes Theta dip at l=1..3 and recover by l=8..14), so an early
        stop at the first decline would always return l = 0 (DSpark paper Sec 5.2 notes
        non-smooth SPS curves need engineering adaptations). Output-equivalence: with
        greedy verification the accepted prefix is a deterministic match, so the window
        choice cannot alter decoded text; for sampled decoding revert to early stopping.
        """
        import torch as _t
        sps = self._sts_sps
        temps = self._sts_temps
        assert sps and temps, "_theta_window called before STS tables loaded"
        keys = sorted(sps)

        def t_round(b: int) -> float:
            if b <= keys[0]: return sps[keys[0]]
            if b >= keys[-1]: return sps[keys[-1]]
            lo = max(k for k in keys if k <= b)
            hi = min(k for k in keys if k >= b)
            if lo == hi: return sps[lo]
            w = (b - lo) / (hi - lo)
            return sps[lo] * (1 - w) + sps[hi] * w

        import math as _math
        block = conf.shape[-1] - 1
        best_l, best_theta = 0, 1.0 / t_round(1)     # l = 0: plain decode step
        a, sum_a = 1.0, 0.0
        conf_row = conf[0].tolist()                  # single device sync for the whole loop
        for k in range(1, block + 1):
            c = min(max(conf_row[k], 1e-6), 1.0 - 1e-6)
            lg = _math.log(c / (1.0 - c))
            p = 1.0 / (1.0 + _math.exp(-lg / temps[k]))   # STS-calibrated conditional
            a *= p
            sum_a += a
            theta = (1.0 + sum_a) / t_round(k + 1)
            if theta > best_theta:
                best_theta, best_l = theta, k
        return best_l
