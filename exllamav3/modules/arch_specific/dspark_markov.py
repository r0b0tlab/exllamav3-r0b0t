"""
Sequential heads of the Qwen3 DSpark draft (Qwen3DSparkModel): the Markov transition head and the confidence head.

Semantics follow the reference implementation (vLLM qwen3_dspark.py and its speculator, Apache-2.0). Every draft row k
yields base logits from the target's LM head; the Markov head adds a bias that depends on the token chosen for the
previous row (row 0 uses the anchor token):

    adj_k(b) = base_k(b) + <markov_w1[x_{k-1}], markov_w2[b]>,    x_k = argmax_b adj_k(b),    x_{-1} = anchor

The bias is evaluated for the top-K candidates of the BASE logits of each row only (candidates are selected once per
row, from the base logits, never from the adjusted ones). That is exactly the selector walk of the DFlash2 draft with
pred = markov_w1, succ = markov_w2, unary = base top-K logits and an all-ones gate, so the existing native kernels
(dflash2_topk, dflash2_selector_walk) run it; a torch formulation is kept for CPU and for the K values the native
top-k does not cover.

The confidence head is sigmoid(proj([h_k ; markov_w1[x_{k-1}]])) over the final-normed draft state h_k: the estimated
probability that draft k is accepted given that drafts 0..k-1 were. It never influences which tokens are drafted.
"""

from __future__ import annotations
from typing_extensions import override
import torch

from ...model.config import Config
from ...util.tensor import to2
from .. import Module

from ...ext import exllamav3_ext as ext


_NATIVE_TOPK = (8, 16, 32)


def _native_topk_ok(logits: torch.Tensor, top_k: int) -> bool:
    """dflash2_topk preconditions: fp16/fp32, unit stride along the vocab dim, 16-byte aligned rows, K in {8, 16, 32}"""
    if not logits.is_cuda or top_k not in _NATIVE_TOPK or logits.dtype not in (torch.half, torch.float):
        return False
    es = logits.element_size()
    return (
        logits.stride(-1) == 1 and
        logits.data_ptr() % 16 == 0 and
        (logits.stride(0) * es) % 16 == 0 and
        (logits.stride(1) * es) % 16 == 0
    )


def markov_candidates(
    logits: torch.Tensor,
    vocab_size: int,
    top_k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Top-K candidates of each row of the base logits, columns beyond vocab_size (a padded head) excluded.

    :param logits:
        (bsz, rows, >= vocab_size) fp16 or fp32

    :return:
        (values (bsz, rows, K) fp32, ids (bsz, rows, K) int64), each row in descending order of value
    """
    bsz, rows = logits.shape[:2]
    if _native_topk_ok(logits, top_k):
        values = torch.empty((bsz, rows, top_k), dtype = torch.float, device = logits.device)
        ids = torch.empty((bsz, rows, top_k), dtype = torch.long, device = logits.device)
        ext.dflash2_topk(logits, vocab_size, 1.0, 0.0, values, ids)
        return values, ids
    values, ids = torch.topk(logits[..., :vocab_size], top_k, dim = -1, sorted = True)
    return values.float(), ids


def markov_walk(
    unary: torch.Tensor,
    cands: torch.Tensor,
    anchor: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    gate: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Left-to-right greedy walk over the candidates, ties to the first candidate in the given order.

    :param unary:
        (bsz, rows, K) fp32 base logits of the candidates

    :param cands:
        (bsz, rows, K) int64 candidate token ids

    :param anchor:
        (bsz,) int64 token that precedes row 0

    :param w1:
        (vocab, rank) fp16/bf16 predecessor table, w2: (vocab, rank) successor table

    :param gate:
        optional (bsz, rows, rank) fp16 scale for the predecessor embedding, all ones when omitted (the DSpark head).
        Only the native path needs it materialized

    :return:
        (bsz, rows + 1) int64 [anchor, x_0, ..., x_{rows-1}]
    """
    bsz, rows, k = unary.shape
    rank = w1.shape[1]
    native = (
        unary.is_cuda and
        w1.dtype in (torch.half, torch.bfloat16) and w1.dtype == w2.dtype and
        w1.is_contiguous() and w2.is_contiguous()
    )
    anchor = anchor.long()
    if native:
        if gate is None:
            gate = torch.ones((bsz, rows, rank), dtype = torch.half, device = unary.device)
        out = torch.empty((bsz, rows + 1), dtype = torch.long, device = unary.device)
        ext.dflash2_selector_walk(
            unary.contiguous(), cands.contiguous(), gate.contiguous(), w1, w2,
            anchor.to(unary.device, non_blocking = anchor.is_pinned()).contiguous(), out, None,
        )
        return out
    pred = anchor
    path = [anchor]
    for i in range(rows):
        a = w1[pred].float()
        if gate is not None:
            a = a * gate[:, i].float()
        b = w2[cands[:, i]].float()
        score = unary[:, i] + torch.einsum("br,bkr->bk", a, b)
        best = torch.argmax(score, dim = -1)
        pred = cands[:, i].gather(-1, best[:, None])[:, 0]
        path.append(pred)
    return torch.stack(path, dim = 1)


def markov_confidence(
    state: torch.Tensor,
    prev_ids: torch.Tensor,
    w1: torch.Tensor,
    conf_w: torch.Tensor,
    conf_b: torch.Tensor,
) -> torch.Tensor:
    """
    sigmoid(proj([h_k ; w1[x_{k-1}]])) in fp32.

    :param state:
        (bsz, rows, hidden) final-normed draft state

    :param prev_ids:
        (bsz, rows) int64 token preceding each row: [anchor, x_0, ..., x_{rows-2}]

    :param conf_w:
        (hidden + rank,) fp32, conf_b: scalar fp32

    :return:
        (bsz, rows) fp32 in (0, 1)
    """
    x = torch.cat((state.float(), w1[prev_ids].float()), dim = -1)
    return torch.sigmoid(x @ conf_w + conf_b)


class DSparkHeads(Module):
    """
    Holder for the raw (unquantized, bf16) tensors of the DSpark heads, so loading, autosplit and compilation account
    for them. Identity in the module walk: sample_from_state runs the heads after the target's LM head.

        markov_head.markov_w1.weight       [vocab, rank]
        markov_head.markov_w2.weight       [draft_vocab, rank]
        confidence_head.proj.weight        [1, hidden + rank]   (only with the confidence head enabled)
        confidence_head.proj.bias          [1]
    """

    def __init__(
        self,
        config: Config,
        key: str,
        vocab_size: int,
        rank: int,
        confidence_in: int | None,
    ):
        super().__init__(config, key, None)
        self.module_name = "DSparkHeads"
        self.vocab_size = vocab_size
        self.rank = rank
        self.confidence_in = confidence_in

        self.key_w1 = f"{key}.markov_w1.weight"
        self.key_w2 = f"{key}.markov_w2.weight"
        self.key_conf_w = "confidence_head.proj.weight"
        self.key_conf_b = "confidence_head.proj.bias"

        self.w1 = None
        self.w2 = None
        self.conf_w = None
        self.conf_b = None
        self._conf_w_src = None
        self._conf_b_src = None
        self.caps.update({"x_cpu": True})

    def optimizer_targets(self):
        return []

    @override
    def weights_numel(self):
        n = 2 * self.vocab_size * self.rank
        if self.confidence_in:
            n += self.confidence_in + 1
        return n + super().weights_numel()

    @property
    def has_confidence(self) -> bool:
        return self.conf_w is not None

    @override
    def can_defer_load(self):
        # load() derives tensors (dtype/layout normalization) from what it reads: the data has to be there
        return False

    @override
    def load(self, device: torch.device, **kwargs):
        super().load(device, **kwargs)
        stc = self.config.stc
        self.w1 = stc.get_tensor(self.key_w1, self.device, optional = False, allow_bf16 = True)
        self.w2 = stc.get_tensor(self.key_w2, self.device, optional = False, allow_bf16 = True)
        expected = (self.vocab_size, self.rank)
        for k, t in ((self.key_w1, self.w1), (self.key_w2, self.w2)):
            if tuple(t.shape) != expected:
                raise ValueError(f"Expected {k} shape {expected}, got {tuple(t.shape)}")
        # The native walk wants one dtype for both tables
        if self.w2.dtype != self.w1.dtype:
            self.w2 = self.w2.to(self.w1.dtype)
        self.w1 = self.w1.contiguous()
        self.w2 = self.w2.contiguous()
        if self.confidence_in:
            w = stc.get_tensor(self.key_conf_w, self.device, optional = False, allow_bf16 = True)
            b = stc.get_tensor(self.key_conf_b, self.device, optional = False, allow_bf16 = True)
            if w.numel() != self.confidence_in:
                raise ValueError(f"Expected {self.key_conf_w} with {self.confidence_in} elements, got {w.numel()}")
            self.conf_w = w.reshape(-1).float().contiguous()
            self.conf_b = b.reshape(()).float()
            self._conf_w_src = w
            self._conf_b_src = b

    @override
    def unload(self):
        self.w1 = None
        self.w2 = None
        self.conf_w = None
        self.conf_b = None
        self._conf_w_src = None
        self._conf_b_src = None
        super().unload()

    @override
    def get_tensors(self):
        t = super().get_tensors()
        if self.w1 is not None:
            t[self.key_w1] = self.w1.contiguous()
            t[self.key_w2] = self.w2.contiguous()
        if self.conf_w is not None:
            t[self.key_conf_w] = self._conf_w_src.contiguous()
            t[self.key_conf_b] = self._conf_b_src.contiguous()
        return t

    @override
    def forward(self, x: torch.Tensor, params: dict, out_dtype = None):
        return to2(x, out_dtype, None)


def adaptive_window(
    conf: torch.Tensor,
    verify_cost: list[float],
    draft_cost: float = 0.0,
) -> int:
    """
    Number of drafts to verify so that the expected accepted tokens per unit of time is highest.

    The expected tokens of a round that verifies n drafts are 1 + sum_{j <= n} prod_{i <= j} conf_i (the verified
    token plus the survival probability of each kept draft), its cost is draft_cost + verify_cost[n]. Requests of a
    batch share one window, the largest any of them asks for (each request still stops at its first mismatch).

    :param conf:
        (bsz, rows) fp32 confidences of the drafts, in order

    :param verify_cost:
        cost of verifying 1 .. rows + 1 rows (anchor + n drafts), index n; any unit

    :param draft_cost:
        cost of producing the drafts, same unit
    """
    bsz, rows = conf.shape
    assert len(verify_cost) >= rows + 1, "verify_cost must cover 1 .. rows + 1 verified rows"
    survival = torch.cumprod(conf.float(), dim = 1)
    expected = torch.cat((torch.ones((bsz, 1), device = conf.device), 1.0 + torch.cumsum(survival, dim = 1)), dim = 1)
    cost = torch.tensor([draft_cost + c for c in verify_cost[:rows + 1]], dtype = torch.float, device = conf.device)
    best = torch.argmax(expected / cost[None, :], dim = 1)
    return int(best.max().item())
