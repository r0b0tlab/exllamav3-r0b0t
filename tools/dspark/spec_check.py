#!/usr/bin/env python3
"""
Speculative decoding checks and measurements for a DSpark draft on a real target pack (one GPU).

Tasks (--tasks, comma separated):

  bench     Per prompt: plain decode and drafted decode (greedy, one stream), tokens, time, accepted length. Arms:
              plain      plain decoding
              plainc     plain decoding with the prompt prefilled in 64-token chunks (a numerical control)
              k<K>       drafted, Markov walk over the top-K candidates (K = 64 is the default of the model)
              k<K>c<t>   ... verifying only the leading drafts whose confidence is at least t
              k<K>a      ... verifying the number of drafts that maximizes the expected tokens per unit of time
                         (cost model measured by the profile task at first use)
            The token streams of every arm are compared with the plain arm: greedy speculative decoding must
            reproduce plain decoding up to floating point noise, every divergence is listed with the target's own
            margin at the first differing position.
  profile   Wall-clock cost of a verify pass over 1 .. 16 rows and of the draft side of a round.
  forced    Drafted decoding where the proposals are replaced by the plain stream with a mismatch forced after
            exactly k tokens, k cycling over 0..15: every partial-window commit (and the full window) is exercised
            by the generator, and the stream must still equal plain decoding.
  rollback  Model level: after a verify window of 16 rows is committed with only k drafts kept (k = 0..15), the
            target recurrent state, target K/V, draft K/V and the next block's drafts are compared with the same
            tokens run one at a time (and with a window that was exactly k + 1 rows long).
  sampled   Sampled decoding: distribution of the first two generated tokens over many samples drawn with the draft
            active, against the target's own distribution (chi-square, KL).

    python tools/dspark/spec_check.py --target <target pack> --draft <draft pack> --prompts <manifest.json> \\
        --tasks bench,forced,rollback,sampled --out result.json

The prompt manifest is {"rows": [{"name", "class", "ids", "max_new"}]} (token ids already rendered with the chat
template). Hold your GPU lock around the run: it loads the whole target.
"""
import argparse
import json
import math
import os
import sys
import time
from types import SimpleNamespace

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

import torch

from exllamav3 import Cache, Config, Generator, Job, Model, Tokenizer
from exllamav3.generator.sampler import ArgmaxSampler, CategoricalSampler

STOP = (248046, 248044)


# ----------------------------------------------------------------------------------------------------------- helpers


def run_job(gen, ids, max_new, sampler = None, stop = STOP, ident = 0, **kw):
    """One job to completion on an otherwise idle generator: (token list, final result, job)"""
    job = Job(
        input_ids = torch.tensor([ids], dtype = torch.long),
        max_new_tokens = max_new,
        stop_conditions = list(stop),
        sampler = sampler or ArgmaxSampler(),
        identifier = ident,
        decode_special_tokens = True,
        **kw
    )
    gen.enqueue(job)
    toks, final = [], None
    while gen.num_remaining_jobs():
        for r in gen.iterate():
            if r.get("identifier") != ident:
                continue
            for src in (r, r.get("held") or {}):
                if "token_ids" in src:
                    toks.extend(int(x) for x in src["token_ids"].flatten().tolist())
            if r.get("eos"):
                final = r
    return toks, final, job


def get_logits(rig, ids):
    """Target logits (fp32 [V]) for the token after ids: a prefill and one single-row step, as plain decoding does"""
    return rig.logits(ids)


def rounds_of(job, new_tokens):
    """(rounds, tokens emitted by drafted rounds): plain steps taken while drafting was skipped count as rounds"""
    st = job.draft_stats
    drafted = sum(a + 1 for _, _, a in st)
    return len(st) + max(0, new_tokens - drafted), drafted


def summarize(rows):
    out = {}
    for cls in sorted({r["class"] for r in rows}) + ["ALL"]:
        sel = [r for r in rows if cls == "ALL" or r["class"] == cls]
        if not sel:
            continue
        tok = sum(r["new_tokens"] for r in sel)
        t = sum(r["time_generate"] for r in sel)
        rounds = sum(r["rounds"] for r in sel)
        out[cls] = {
            "prompts": len(sel), "new_tokens": tok, "tok_s": tok / t if t else None,
            "AL": tok / rounds if rounds else None,
            "AL_mean_per_prompt": sum(r["new_tokens"] / r["rounds"] for r in sel) / len(sel),
            "mean_tok_s_per_prompt": sum(r["new_tokens"] / r["time_generate"] for r in sel) / len(sel),
        }
    return out


def chi_square(counts, probs, min_expected = 5.0):
    """Pearson chi-square of observed counts against probabilities; cells with a small expectation are pooled.
    Returns (statistic, dof, p_value (Wilson-Hilferty), KL(emp || p) in nats, expected KL under the null)"""
    n = float(counts.sum())
    exp = probs * n
    order = torch.argsort(exp, descending = True)
    big = order[exp[order] >= min_expected]
    small = order[exp[order] < min_expected]
    o = counts[big].double().tolist()
    e = exp[big].double().tolist()
    if len(small):
        o.append(float(counts[small].sum()))
        e.append(float(exp[small].sum()))
    stat = sum((a - b) ** 2 / b for a, b in zip(o, e) if b > 0)
    dof = max(1, len(o) - 1)
    z = ((stat / dof) ** (1 / 3) - (1 - 2 / (9 * dof))) / math.sqrt(2 / (9 * dof))
    pval = 0.5 * math.erfc(z / math.sqrt(2))
    emp = counts.double() / n
    nz = emp > 0
    kl = float((emp[nz] * (emp[nz] / probs.double()[nz].clamp_min(1e-300)).log()).sum())
    return stat, dof, pval, kl, dof / (2 * n)


# ----------------------------------------------------------------------------------------------------------- bench


def arm_settings(arm: str):
    """plain | plainc (plain, prompt prefilled in 64-token chunks: a numerical control) |
    k<K> (Markov top-K) with optionally c<thr> (confidence threshold) or a (adaptive verification)"""
    if arm in ("plain", "plainc"):
        return None, None, False
    assert arm.startswith("k")
    body = arm[1:]
    if body.endswith("a"):
        return int(body[:-1]), None, True
    k, _, c = body.partition("c")
    return int(k), (float(c) if c else None), False


def run_arm(arm, rows, models, warmup = 2):
    k, thr, adaptive = arm_settings(arm)
    draft = models.draft
    gen = models.gen_plain if arm == "plain" else models.gen_plain_chunk if arm == "plainc" else models.gen_spec
    if k is not None:
        draft.markov_top_k = k
        draft.draft_conf_threshold = thr
        draft.adaptive_verification = adaptive
        if adaptive:
            ensure_profile(models)
    for r in rows[:warmup]:
        run_job(gen, r["ids"], min(r["max_new"], 96))
    out = []
    for r in rows:
        toks, final, job = run_job(gen, r["ids"], r["max_new"])
        n = final["new_tokens"]
        rounds, drafted = rounds_of(job, n) if k is not None else (n, 0)
        out.append({
            "name": r["name"], "class": r["class"], "new_tokens": n, "time_generate": final["time_generate"],
            "time_prefill": final["time_prefill"], "rounds": rounds, "tokens": toks,
        })
    return out


def margin_report(rig, prompts_by_name, plain, other):
    """First differing position of each diverging prompt with the target's own logit gap between the two tokens"""
    rep = []
    for p, o in zip(plain, other):
        a, b = p["tokens"], o["tokens"]
        n = min(len(a), len(b))
        j = next((i for i in range(n) if a[i] != b[i]), None)
        if j is None:
            if len(a) != len(b):
                rep.append({"name": p["name"], "first_diff": n, "note": "one stream is a prefix of the other",
                            "len_plain": len(a), "len_arm": len(b)})
            continue
        ids = prompts_by_name[p["name"]] + a[:j]
        lg = get_logits(rig, ids)
        la, lb = float(lg[a[j]]), float(lg[b[j]])
        top2 = torch.topk(lg, 2)
        rep.append({"name": p["name"], "class": p["class"], "first_diff": j, "token_plain": a[j], "token_arm": b[j],
                    "logit_gap_nats": la - lb, "top2_gap_nats": float(top2.values[0] - top2.values[1]),
                    "plain_is_top1": a[j] == int(top2.indices[0])})
    return rep


def task_bench(args, models, rows, result):
    target, draft, gen_plain, gen_spec = models.target, models.draft, models.gen_plain, models.gen_spec
    by_name = {r["name"]: r["ids"] for r in rows}
    res = {}
    out = {"summary": {}, "divergence": {}, "errors": {}}
    if args.keep_tokens:
        out["tokens"] = {}
    result["bench"] = out
    reference = None
    if args.reference_tokens:
        ref = json.load(open(args.reference_tokens))["bench"]["tokens"]["plain"]
        assert len(ref) == len(rows), "reference token streams do not match the prompt list"
        reference = [{"name": r_["name"], "class": r_["class"], "tokens": t} for r_, t in zip(rows, ref)]
    for arm in args.arms.split(","):
        t0 = time.time()
        try:
            r = run_arm(arm, rows, models)
        except Exception as e:
            # an arm that cannot run must not cost the results of the others
            import traceback
            traceback.print_exc()
            out["errors"][arm] = repr(e)
            continue
        res[arm] = r
        s = summarize(r)
        out["summary"][arm] = s
        if args.keep_tokens:
            out["tokens"][arm] = [r_["tokens"] for r_ in r]
        print(f"[bench] arm {arm:10s} {time.time() - t0:6.0f}s  " +
              "  ".join(f"{c}: {v['tok_s']:.1f} tok/s AL {v['AL']:.2f}" for c, v in s.items()), flush = True)
        plain = res.get("plain") or reference
        if plain and arm != "plain":
            same = sum(1 for p_, o in zip(plain, r) if p_["tokens"] == o["tokens"])
            out["divergence"][arm] = {"identical": same, "of": len(r),
                                      "diverging": margin_report(models.rig, by_name, plain, r)}
            print(f"[bench] arm {arm}: {same}/{len(r)} streams identical to plain; diverging: "
                  f"{json.dumps(out['divergence'][arm]['diverging'])}", flush = True)
        if args.out:
            json.dump(result, open(args.out, "w"), indent = 1)


# ----------------------------------------------------------------------------------------------------------- forced


def task_forced(args, models, rows, result):
    target, draft, gen_plain, gen_spec = models.target, models.draft, models.gen_plain, models.gen_spec
    vocab = target.config.vocab_size
    draft.markov_top_k = 64
    draft.draft_conf_threshold = None
    draft.adaptive_verification = False
    schedule = list(range(16)) + list(range(15, -1, -1))
    res = []
    orig = draft.sample_from_state
    for r in rows[::13][:args.forced_prompts]:
        cont, _, _ = run_job(gen_plain, r["ids"], r["max_new"])
        truth = list(r["ids"]) + cont
        st = {"round": 0, "k": []}

        def forced(state, params, st = st, truth = truth):
            out = orig(state, params)
            P = int(params["cache_seqlens"][0])
            k = schedule[st["round"] % len(schedule)]
            st["round"] += 1
            st["k"].append(k)
            ids = [int(out[0, 0])]
            for j in range(15):
                pos = P + 1 + j
                t = truth[pos] if pos < len(truth) else (ids[0] + 1) % vocab
                if j >= k:
                    t = (t + 1 + j) % vocab
                ids.append(t)
            return torch.tensor([ids], dtype = torch.long, device = out.device)

        draft.sample_from_state = forced
        try:
            toks, final, job = run_job(gen_spec, r["ids"], r["max_new"])
        finally:
            draft.sample_from_state = orig
        acc = [a for _, _, a in job.draft_stats]
        want = st["k"][:len(acc)]
        # the last round may be cut short by the token limit or EOS
        ok_acc = all(a == w for a, w in zip(acc[:-1], want[:-1]))
        res.append({"name": r["name"], "tokens_identical": toks == cont, "len": len(cont), "rounds": len(acc),
                    "accepted_matches_schedule": ok_acc,
                    "first_diff": next((i for i in range(min(len(toks), len(cont))) if toks[i] != cont[i]), None),
                    "keep_values_seen": sorted(set(acc))})
        print(f"[forced] {r['name']}: identical={res[-1]['tokens_identical']} rounds={len(acc)} "
              f"schedule_ok={ok_acc} keep_seen={res[-1]['keep_values_seen']}", flush = True)
    result["forced"] = res


# ----------------------------------------------------------------------------------------------------------- rollback


def snapshot(cache, dcache, state, block_table, n_rows):
    """Recurrent state, target K/V and draft K/V of positions [0, n_rows) of one sequence"""
    pages = block_table[0].tolist()
    pos = torch.arange(n_rows)
    flat = torch.tensor([pages[p // 256] * 256 + p % 256 for p in pos.tolist()], dtype = torch.long)
    snap = {"gdn_conv": {}, "gdn_state": {}, "kv": {}, "dkv": {}}
    cdim = None
    for inst, layer in cache.recurrent_layers.items():
        cdim = layer.module.conv_kernel_size
        snap["gdn_conv"][inst] = layer.conv_state[state.slot, :, :cdim].float().cpu().clone()
        snap["gdn_state"][inst] = layer.recurrent_state[state.slot, 0].float().cpu().clone()
    for inst, layer in cache.layers.items():
        d = layer.k.device
        k = layer.k.reshape(-1, *layer.k.shape[2:])[flat.to(d)].float().cpu()
        v = layer.v.reshape(-1, *layer.v.shape[2:])[flat.to(d)].float().cpu()
        snap["kv"][inst] = (k, v)
    for inst, layer in dcache.layers.items():
        d = layer.k.device
        k = layer.k.reshape(-1, *layer.k.shape[2:])[flat.to(d)].float().cpu()
        v = layer.v.reshape(-1, *layer.v.shape[2:])[flat.to(d)].float().cpu()
        snap["dkv"][inst] = (k, v)
    return snap


def _rel(a, b):
    a, b = a.double(), b.double()
    return float((a - b).norm() / b.norm().clamp_min(1e-30)), float((a - b).abs().max())


def compare_snapshots(a, b):
    """Worst relative L2 / max abs over layers for each component"""
    out = {}
    for comp in ("gdn_conv", "gdn_state"):
        r = [_rel(a[comp][i], b[comp][i]) for i in a[comp]]
        out[comp] = {"rel_l2": max(x for x, _ in r), "max_abs": max(y for _, y in r)}
    for comp in ("kv", "dkv"):
        r = [_rel(torch.cat([a[comp][i][0].flatten(), a[comp][i][1].flatten()]),
                  torch.cat([b[comp][i][0].flatten(), b[comp][i][1].flatten()])) for i in a[comp]]
        out[comp] = {"rel_l2": max(x for x, _ in r), "max_abs": max(y for _, y in r)}
    return out


class Rig:
    """Direct model-level driving (prefill, windows, draft updates) on private caches, one sequence per page range"""

    def __init__(self, target, draft, cache, dcache):
        self.target, self.draft, self.cache, self.dcache = target, draft, cache, dcache

    def params_for(self, table, seqlen, state, history = False):
        p = {
            "attn_mode": "flash_attn", "block_table": table, "cache": self.cache,
            "cache_seqlens": torch.tensor([seqlen], dtype = torch.int32),
            "recurrent_states": [state], "indexed_embeddings": [], "recurrent_history": history,
        }
        p.update(self.draft.draft_verifier_params)
        return p

    def draft_update(self, p, table, seqlen, lengths = None):
        self.draft.update_kv_from_target(
            target_hidden = [t.clone() for t in p["export_states"]], cache = self.dcache,
            params = {"block_table": table, "cache_seqlens": torch.tensor([seqlen], dtype = torch.int32)},
            lengths = lengths,
        )

    @torch.inference_mode()
    def prefill(self, ids, table, state):
        pos = 0
        while pos < len(ids):
            chunk = torch.tensor([ids[pos:pos + 256]], dtype = torch.long)
            p = self.params_for(table, pos, state)
            self.target.prefill(chunk, p)
            self.draft_update(p, table, pos)
            pos += chunk.shape[1]

    @torch.inference_mode()
    def window(self, ids, table, state, start, n, history):
        p = self.params_for(table, start, state, history)
        logits = self.target.forward(torch.tensor([ids[start:start + n]], dtype = torch.long), p)
        return p, logits

    @torch.inference_mode()
    def next_drafts(self, table, seqlen, anchor):
        pp = {"attn_mode": "flash_attn", "block_table": table, "cache": self.dcache,
              "cache_seqlens": torch.tensor([seqlen], dtype = torch.int32)}
        st = self.draft.forward(torch.tensor([[anchor]], dtype = torch.long), pp)
        ids = self.draft.sample_from_state(st, pp)
        return ids[0].cpu(), st.float().cpu()

    @torch.inference_mode()
    def logits(self, ids, page0 = 16):
        table = torch.arange(page0, page0 + 4, dtype = torch.int32)[None]
        s = self.cache.get_new_state()
        try:
            self.prefill(ids[:-1], table, s)
            _, lg = self.window(ids, table, s, len(ids) - 1, 1, False)
            return lg[0, -1].float().cpu()
        finally:
            s.free()


@torch.inference_mode()
def task_rollback(args, models, rows, result):
    target, draft, gen_plain, rig = models.target, models.draft, models.gen_plain, models.rig
    cache, dcache = rig.cache, rig.dcache
    vocab = target.config.vocab_size
    # Independent page ranges: A (16 rows, commit anchor + keep drafts), J (the same with different tokens after the
    # kept ones), D (exactly keep + 1 rows), B (one token at a time)
    bt = {n: torch.arange(i * 4, i * 4 + 4, dtype = torch.int32)[None] for i, n in enumerate("AJDB")}

    draft.markov_top_k = 64
    draft.draft_conf_threshold = None
    draft.adaptive_verification = False
    # a long text sequence: a prompt followed by its own plain continuation (the longest of the first few)
    seq = []
    for r in rows[:4]:
        cont, _, _ = run_job(gen_plain, r["ids"], 320)
        if len(r["ids"]) + len(cont) > len(seq):
            seq = list(r["ids"]) + cont
    res = []
    # committed lengths: a short one, and one whose window straddles a page boundary of the paged cache
    lens = []
    for L in (int(x) for x in args.rollback_lengths.split(",")):
        L = min(L, len(seq) - 17, 4 * 256 - 40)
        if L > 0 and L not in lens:
            lens.append(L)
    print(f"[rollback] sequence of {len(seq)} tokens, committed lengths {lens}", flush = True)
    for L in lens:
        for keep in range(16):
            states = {n: cache.get_new_state() for n in "AJDB"}
            try:
                sA, sJ, sD, sB = (states[n] for n in "AJDB")
                # A: 16-row window, then commit anchor + keep drafts (the generator's rejection path)
                rig.prefill(seq[:L], bt["A"], sA)
                pA, _ = rig.window(seq, bt["A"], sA, L, 16, True)
                sA.rewind(15 - keep)
                rig.draft_update(pA, bt["A"], L, lengths = [keep + 1])
                # J: the drafts after the kept ones replaced by other tokens: the committed state must not notice
                junk = seq[:L + 1 + keep] + [(t + 7919 + i) % vocab for i, t in enumerate(seq[L + 1 + keep:L + 16])]
                rig.prefill(seq[:L], bt["J"], sJ)
                pJ, _ = rig.window(junk, bt["J"], sJ, L, 16, True)
                sJ.rewind(15 - keep)
                rig.draft_update(pJ, bt["J"], L, lengths = [keep + 1])
                # D: a window of exactly keep + 1 rows (normalized as a fully accepted window)
                rig.prefill(seq[:L], bt["D"], sD)
                pD, _ = rig.window(seq, bt["D"], sD, L, keep + 1, True)
                sD.rewind(0)
                rig.draft_update(pD, bt["D"], L)
                # B: the same tokens one at a time
                rig.prefill(seq[:L], bt["B"], sB)
                for j in range(keep + 1):
                    pB, _ = rig.window(seq, bt["B"], sB, L + j, 1, False)
                    rig.draft_update(pB, bt["B"], L + j)
                n = L + keep + 1
                assert sA.position == sJ.position == sD.position == sB.position == n, \
                    (sA.position, sJ.position, sD.position, sB.position, n)
                snaps = {k: snapshot(cache, dcache, states[k], bt[k], n) for k in "AJDB"}
                nd = {k: rig.next_drafts(bt[k], n, seq[n]) for k in "AJDB"}
                entry = {
                    "L": L, "keep": keep,
                    "A_vs_J": compare_snapshots(snaps["A"], snaps["J"]),
                    "A_vs_D": compare_snapshots(snaps["A"], snaps["D"]),
                    "A_vs_B": compare_snapshots(snaps["A"], snaps["B"]),
                    "D_vs_B": compare_snapshots(snaps["D"], snaps["B"]),
                    "next_draft_state_rel_l2_A_vs_J": _rel(nd["A"][1], nd["J"][1])[0],
                    "next_draft_state_rel_l2_A_vs_B": _rel(nd["A"][1], nd["B"][1])[0],
                    "next_drafts_equal_A_J": int((nd["A"][0] == nd["J"][0]).sum()),
                    "next_drafts_equal_A_B": int((nd["A"][0] == nd["B"][0]).sum()),
                }
                res.append(entry)

                def f(c, comp):
                    return f"{entry[c][comp]['rel_l2']:.1e}"
                print(f"[rollback] L={L} keep={keep:2d} | A~J conv {f('A_vs_J', 'gdn_conv')} state {f('A_vs_J', 'gdn_state')} "
                      f"kv {f('A_vs_J', 'kv')} dkv {f('A_vs_J', 'dkv')} drafts {entry['next_drafts_equal_A_J']}/16 | "
                      f"A~B conv {f('A_vs_B', 'gdn_conv')} state {f('A_vs_B', 'gdn_state')} kv {f('A_vs_B', 'kv')} "
                      f"dkv {f('A_vs_B', 'dkv')} drafts {entry['next_drafts_equal_A_B']}/16 | "
                      f"A~D state {f('A_vs_D', 'gdn_state')} D~B state {f('D_vs_B', 'gdn_state')}", flush = True)
            finally:
                for s in states.values():
                    s.free()
    result["rollback"] = res
    comps = ("gdn_conv", "gdn_state", "kv", "dkv")
    result["rollback_summary"] = {
        k: {c: max(e[k][c]["rel_l2"] for e in res) for c in comps}
        for k in ("A_vs_J", "A_vs_B", "A_vs_D", "D_vs_B")
    }
    print("[rollback] summary (worst rel_l2 over all cases)", json.dumps(result["rollback_summary"]), flush = True)


# ----------------------------------------------------------------------------------------------------------- profile


@torch.inference_mode()
def ensure_profile(models, context = 300, reps = 7):
    """Cost of one verify pass over 1 .. 16 rows and of the draft side of a round (milliseconds, wall clock, one
    stream at `context` committed tokens); installed as the model's cost model for adaptive verification"""
    if getattr(models, "profile", None) is not None:
        return models.profile
    rig, draft = models.rig, models.draft
    ids = models.profile_ids
    assert len(ids) >= context + 17
    table = torch.arange(16, 20, dtype = torch.int32)[None]
    state = rig.cache.get_new_state()
    verify = []
    try:
        rig.prefill(ids[:context], table, state)
        p = None
        for n in range(1, 17):
            ts = []
            for rep in range(reps + 2):
                state.position = context
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                p, _ = rig.window(ids, table, state, context, n, True)
                torch.cuda.synchronize()
                ts.append(time.perf_counter() - t0)
            ts = sorted(ts[2:])
            verify.append(1000 * ts[len(ts) // 2])
        state.position = context
        ts, tu = [], []
        for rep in range(reps + 2):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            rig.next_drafts(table, context, ids[context])
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            rig.draft_update(p, table, context, lengths = [16])
            torch.cuda.synchronize()
            ts.append(t1 - t0)
            tu.append(time.perf_counter() - t1)
        ts, tu = sorted(ts[2:]), sorted(tu[2:])
        draft_ms, update_ms = 1000 * ts[len(ts) // 2], 1000 * tu[len(tu) // 2]
    finally:
        state.free()
    draft.set_verify_cost(verify, draft_ms + update_ms)
    models.profile = {"context": context, "verify_ms": verify, "draft_ms": draft_ms, "draft_kv_update_ms": update_ms}
    print(f"[profile] verify ms by rows 1..16: {[round(v, 1) for v in verify]}; draft {draft_ms:.1f} ms, "
          f"draft K/V update {update_ms:.1f} ms", flush = True)
    return models.profile


def task_profile(args, models, rows, result):
    result["profile"] = ensure_profile(models)


# ----------------------------------------------------------------------------------------------------------- sampled


def task_sampled(args, models, rows, result):
    target, draft, gen_plain, gen_spec = models.target, models.draft, models.gen_plain, models.gen_spec
    draft.markov_top_k = 64
    draft.draft_conf_threshold = None
    draft.adaptive_verification = False
    out = []
    for name in args.sampled_prompts.split(","):
        r = next(x for x in rows if x["name"] == name)
        ids = r["ids"]
        n = args.samples
        # A test needs distributions with several well populated cells: raise the temperature from the requested one
        # until the first token and the token after the most likely first token both have >= 12 cells expecting >= 5
        lg0 = get_logits(models.rig, ids)
        lg1 = get_logits(models.rig, ids + [int(lg0.argmax())])
        T = None
        for t_try in [args.temperature] + [t for t in (1.5, 2.0, 3.0, 4.0) if t > args.temperature]:
            c0 = (torch.softmax(lg0 / t_try, -1) * n >= 5).sum()
            c1 = (torch.softmax(lg1 / t_try, -1) * n * float(torch.softmax(lg0 / t_try, -1).max()) >= 5).sum()
            T = t_try
            if c0 >= 12 and c1 >= 12:
                break
        p0 = torch.softmax(lg0 / T, dim = -1)
        counts1 = torch.zeros(p0.shape[-1], dtype = torch.long)
        firsts, seconds = [], []
        t0 = time.time()
        for i in range(n):
            gen_spec.enqueue(Job(
                input_ids = torch.tensor([ids], dtype = torch.long), max_new_tokens = 2,
                sampler = CategoricalSampler(temperature = T),
                stop_conditions = [], identifier = i, decode_special_tokens = True,
            ))
        toks = {i: [] for i in range(n)}
        while gen_spec.num_remaining_jobs():
            for rr in gen_spec.iterate():
                i = rr.get("identifier")
                if i is None:
                    continue
                for src in (rr, rr.get("held") or {}):
                    if "token_ids" in src:
                        toks[i].extend(int(x) for x in src["token_ids"].flatten().tolist())
        samples = [toks[i][:2] for i in range(n)]
        assert all(len(s) == 2 for s in samples)
        for a, _ in samples:
            counts1[a] += 1
        stat, dof, pval, kl, kl_exp = chi_square(counts1, p0)
        entry = {"prompt": name, "n": n, "temperature": T, "secs": round(time.time() - t0, 1),
                 "first_token": {"chi2": stat, "dof": dof, "p": pval, "kl": kl, "kl_expected_null": kl_exp}}
        # the second token given the most frequent first tokens: drawn after an accepted draft or after a rejection,
        # both must follow the target's conditional distribution
        top_first = torch.topk(counts1, 2).indices.tolist()
        cond = []
        for c in top_first:
            sec = [b for a, b in samples if a == c]
            if len(sec) < 50:
                continue
            p1 = torch.softmax(get_logits(models.rig, ids + [c]) / T, dim = -1)
            cnt = torch.zeros_like(counts1)
            for b in sec:
                cnt[b] += 1
            stat, dof, pval, kl, kl_exp = chi_square(cnt, p1)
            cond.append({"first": c, "n": len(sec), "chi2": stat, "dof": dof, "p": pval, "kl": kl,
                         "kl_expected_null": kl_exp, "p_first_under_target": float(p0[c])})
        entry["second_given_first"] = cond
        out.append(entry)
        print(f"[sampled] {json.dumps(entry)}", flush = True)
    result["sampled"] = out


# ----------------------------------------------------------------------------------------------------------- main


def main():
    ap = argparse.ArgumentParser(description = __doc__.split("\n")[1])
    ap.add_argument("--target", required = True)
    ap.add_argument("--draft", required = True)
    ap.add_argument("--prompts", required = True)
    ap.add_argument("--tasks", default = "bench")
    ap.add_argument("--arms", default = "plain,k64", help = "bench arms, e.g. plain,k64,k32,k16,k64c0.5")
    ap.add_argument("--limit", type = int, default = 0, help = "use the first N prompts of each class (0: all)")
    ap.add_argument("--classes", default = "", help = "comma separated classes to keep")
    ap.add_argument("--device", default = "cuda:0")
    ap.add_argument("--cache-tokens", type = int, default = 8192)
    ap.add_argument("--batch", type = int, default = 4)
    ap.add_argument("--forced-prompts", type = int, default = 6)
    ap.add_argument("--rollback-lengths", default = "40,246")
    ap.add_argument("--sampled-prompts", default = "gsm8k-003,ifeval-004")
    ap.add_argument("--samples", type = int, default = 2000)
    ap.add_argument("--temperature", type = float, default = 1.0)
    ap.add_argument("--keep-tokens", action = "store_true")
    ap.add_argument("--reference-tokens", default = None, help = "result JSON of an earlier run with --keep-tokens: its plain "
                    "arm is the stream the arms are compared with when this run has no plain arm")
    ap.add_argument("--out", default = None)
    args = ap.parse_args()
    device = torch.device(args.device)

    rows = json.load(open(args.prompts))["rows"]
    if args.classes:
        keep = set(args.classes.split(","))
        rows = [r for r in rows if r["class"] in keep]
    if args.limit:
        seen = {}
        sel = []
        for r in rows:
            seen[r["class"]] = seen.get(r["class"], 0) + 1
            if seen[r["class"]] <= args.limit:
                sel.append(r)
        rows = sel

    t0 = time.time()
    tcfg = Config.from_directory(args.target)
    target = Model.from_config(tcfg)
    dcfg = Config.from_directory(args.draft)
    draft = Model.from_config(dcfg)
    # One cache per generator (a generator's page table owns every page of its cache and caches prefixes in them),
    # and one for the model-level rollback comparison
    cache_plain = Cache(target, max_num_tokens = args.cache_tokens, max_batch_size = args.batch)
    cache_spec = Cache(target, max_num_tokens = args.cache_tokens, max_batch_size = args.batch, max_history = 15)
    dcache_spec = Cache(draft, max_num_tokens = args.cache_tokens)
    cache_rb = Cache(target, max_num_tokens = 20 * 256, max_batch_size = 5, max_history = 15)
    dcache_rb = Cache(draft, max_num_tokens = 20 * 256)
    cache_ctl = Cache(target, max_num_tokens = args.cache_tokens, max_batch_size = args.batch) \
        if "plainc" in args.arms else None
    target.load(device, progressbar = False)
    draft.load(device, progressbar = False)
    tok = Tokenizer.from_config(tcfg)
    print(f"loaded in {time.time() - t0:.0f}s", flush = True)

    gen_plain = Generator(model = target, cache = cache_plain, tokenizer = tok, max_batch_size = args.batch)
    gen_plain_chunk = None
    if "plainc" in args.arms:
        # numerical control: the same plain decoding with the prompt prefilled in different chunks
        gen_plain_chunk = Generator(model = target, cache = cache_ctl, tokenizer = tok, max_batch_size = args.batch,
                                    max_chunk_size = 64)
    gen_spec = Generator(model = target, cache = cache_spec, tokenizer = tok, max_batch_size = args.batch,
                         draft_model = draft, draft_cache = dcache_spec, num_draft_tokens = 15, record_draft_stats = True)
    profile_ids = []
    for r in json.load(open(args.prompts))["rows"]:
        profile_ids += r["ids"]
    models = SimpleNamespace(target = target, draft = draft, tok = tok, gen_plain = gen_plain, gen_spec = gen_spec,
                             gen_plain_chunk = gen_plain_chunk, profile = None, profile_ids = profile_ids[:600],
                             rig = Rig(target, draft, cache_rb, dcache_rb))

    result = {"target": os.path.basename(os.path.normpath(args.target)), "draft": os.path.basename(os.path.normpath(args.draft)),
              "prompts": len(rows), "cache_tokens": args.cache_tokens}
    tasks = {"bench": task_bench, "forced": task_forced, "rollback": task_rollback, "sampled": task_sampled,
             "profile": task_profile}
    for t in args.tasks.split(","):
        t1 = time.time()
        try:
            tasks[t](args, models, rows, result)
        except Exception as e:
            # one failing task must not cost the lock-holding time of the others
            import traceback
            traceback.print_exc()
            result[f"{t}_error"] = repr(e)
        print(f"[{t}] done in {time.time() - t1:.0f}s", flush = True)
        if args.out:
            json.dump(result, open(args.out, "w"), indent = 1)


if __name__ == "__main__":
    main()
