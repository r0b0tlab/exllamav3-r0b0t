"""
Opt-in guard against the encoder's global-scale collapse (convert.py --retry_outliers) and the helpers that
dump raw Hessians for a later repair (convert.py --dump_hessians).

Why: `g_scale_gss` (the global scale search in `regularize`) collapses deterministically on some sign-flip draws.
The scale ends at its lower bound (0.01) or far below the 0.35..1.3 of a good draw, the tiles are quantized to noise
and the matrix comes out with a proxy error 20..600x its neighbours' and a weight rel_l2 of 0.8..2.6 against the
usual ~0.09 (observed on ~700 of ~41k expert matrices of one large MoE model, none of them caught by the encoder itself).
`seed = idx` is the module index and is shared by every matrix of a layer, so every expert of a layer gets the same
sign flips (su is drawn once per Hessian, sv is the first draw after the seed); a bad draw therefore hits whichever
matrices happen to be sensitive to it.

This module holds only pure functions (no torch import) so the selection logic can be unit-tested on CPU; the
orchestration (re-encoding, installing the winner) lives in convert_model.py.
"""

from __future__ import annotations
import json
import re
import statistics

# Seed schedule: the original encode used `seed = idx` (k = 0). Retry k uses idx + 1000 * k, k = 1 .. RETRY_MAX_SEEDS.
# The stride keeps retry seeds of one module away from the module indices of the original encodes (idx < 1000).
SEED_STRIDE = 1000
RETRY_MAX_SEEDS = 32

# Defaults of the outlier test: proxy error more than ~5x its group's median,
# or a global scale below ~0.2 (good draws give 0.35..1.3, collapsed ones 0.01..0.19)
DEFAULT_RATIO = 5.0
DEFAULT_MIN_GSCALE = 0.2
# The ratio test needs a population to take a median over (a module's expert matrices, or its dense ones)
MIN_GROUP = 8


def retry_seed(idx: int, k: int) -> int:
    """Seed of retry number k (1-based) for a module with index idx. k = 0 is the original encode's `seed = idx`."""
    assert k >= 0
    return idx + SEED_STRIDE * k


def group_key(key: str, K: int):
    """Peer group of a linear for the median test: same name with every numeric path component wildcarded
    (layers.5.mlp.experts.17.down_proj -> layers.*.mlp.experts.*.down_proj) at the same bitrate."""
    return (re.sub(r"\.\d+(?=\.|$)", ".*", key), K)


def find_outliers(
    results: dict,
    ratio: float = DEFAULT_RATIO,
    min_gscale: float = DEFAULT_MIN_GSCALE,
    min_group: int = MIN_GROUP,
):
    """
    Select the matrices of one module that need a re-encode.

    :param results:
        {linear key: {"proxy_err": float, "g_scale": float, "K": int, "q_fallback": bool, "zeros": bool}}
    :return:
        (outliers, medians): outliers = {key: reason}; medians = {group key: median proxy error} for the groups
        that had enough members for the ratio test.

    Fallback encodes (no Hessian: the number is an rmse, not a proxy error) and all-zero tensors are never selected;
    a proxy error < 0 is the encoder's out-of-memory sentinel and only takes part in the g_scale test.
    """
    groups = {}
    for key, r in results.items():
        if r.get("q_fallback") or r.get("zeros") or r["K"] == 16:
            continue
        pe = r["proxy_err"]
        if pe is not None and pe > 0.0:
            groups.setdefault(group_key(key, r["K"]), []).append(pe)

    medians = {g: statistics.median(v) for g, v in groups.items() if len(v) >= min_group}

    outliers = {}
    for key, r in results.items():
        if r.get("q_fallback") or r.get("zeros") or r["K"] == 16:
            continue
        gs, pe = r["g_scale"], r["proxy_err"]
        reasons = []
        if gs is not None and gs < min_gscale:
            reasons.append(f"g_scale {gs:.4f} < {min_gscale}")
        med = medians.get(group_key(key, r["K"]))
        if med is not None and pe is not None and pe > ratio * med:
            reasons.append(f"proxy_err {pe:.6f} > {ratio:g} x median {med:.6f}")
        if reasons:
            outliers[key] = "; ".join(reasons)
    return outliers, medians


def acceptable(proxy_err, g_scale, median, ratio: float = DEFAULT_RATIO, min_gscale: float = DEFAULT_MIN_GSCALE):
    """The test a retry candidate has to pass (the negation of the outlier test)."""
    if proxy_err is None or proxy_err < 0.0:
        return False
    if g_scale is not None and g_scale < min_gscale:
        return False
    if median is not None and proxy_err > ratio * median:
        return False
    return True


def pick_best(candidates: list):
    """
    The candidate to keep when none was acceptable: lowest proxy error among those with a usable one, preferring a
    candidate whose global scale did not collapse. Each candidate: {"seed", "proxy_err", "g_scale", ...}. Returns an
    index into the list (0 is the original encode, so ties keep it).
    """
    def rank(i):
        c = candidates[i]
        pe = c["proxy_err"]
        usable = pe is not None and pe > 0.0
        return (0 if usable else 1, pe if usable else 0.0, i)
    return min(range(len(candidates)), key = rank)


def summarize(records: list) -> dict:
    """Header block for the pack's retry log."""
    return {
        "retried": len(records),
        "resolved": sum(1 for r in records if r.get("resolved")),
        "unresolved": [r["key"] for r in records if not r.get("resolved")],
        "kept_original": [r["key"] for r in records if r.get("winner", {}).get("seed") == r.get("original", {}).get("seed")],
    }


def read_log(path: str) -> list:
    """Read the jsonl written during conversion. A module re-done after a crash appends again: the last record of a
    key wins."""
    by_key = {}
    with open(path, "r", encoding = "utf8") as f:
        for line in f:
            line = line.strip()
            if line:
                r = json.loads(line)
                by_key[r["key"]] = r
    return list(by_key.values())
