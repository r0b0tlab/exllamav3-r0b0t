#!/usr/bin/env python3
"""
Stage-by-stage numerical gate of the DSpark draft forward against reference dumps.

The reference dump of one anchor is a directory with manifest.json and raw little-endian tensors (float32 / int32), as
written by a torch reference implementation of the semantics of vLLM's qwen3_dspark.py (Apache-2.0). Tensors used here:

    tap_{N}            [T, hidden]        target residual stream after decoder layer N (the draft's input)
    ids                [T]                token ids of the sequence
    anchor_token       [1], anchor_pos [1]  the anchor token and P = number of committed positions (<= T)
    fc, hidden_norm    [T, hidden]        context projection and its norm
    ctx_k{l}, ctx_v{l} [T, kv_heads, head_dim]   context K (normed + RoPE'd) and V of draft layer l
    block_embed        [1, B, hidden]     block row embeddings
    layer{l}_out       [1, B, hidden]     block output of draft layer l
    final_norm         [1, B, hidden]     final norm output
    base_logits        [B, V]             target LM head over the block rows
    tokens_{K}, idx_{K}, adj_{K}, conf_{K}   Markov-walked picks, top-K candidate ids, adjusted candidate logits and
                                          confidence for top-K candidates (K = 16, 32, 64 and "full")

Error per stage is the relative L2 norm ||fork - ref|| / ||ref|| against the float32 reference built from the SAME
quantized draft weights, so what is measured is the fork's activation arithmetic (fp16) and kernels. The tolerance is
calibrated from the floor of lower-precision reference runs of the same stage (--floor name=dir, repeatable): a stage
passes when its error is at most --factor times the largest floor (and at least --min-tol).

The target is only used for its embedding and LM head: the full target is not loaded.

    python tools/dspark/stage_gate.py --draft <draft pack> --target <target pack> --ref <dump dir> \\
        --floor bf16=<dump dir> --floor fp16=<dump dir> [--bf16-head <hf dir>] --out gate.json
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

import torch

from exllamav3 import Cache, Config, Model
from exllamav3.modules.arch_specific.dspark_markov import (
    markov_candidates, markov_walk, markov_confidence,
)
from exllamav3.ext import exllamav3_ext as ext


def read_dump(path: str) -> dict:
    man = json.load(open(os.path.join(path, "manifest.json")))
    out = {}
    for name, info in man["tensors"].items():
        dt = torch.float32 if info["dtype"] == "f32" else torch.int32
        t = torch.frombuffer(bytearray(open(os.path.join(path, info["file"]), "rb").read()), dtype = dt)
        out[name] = t.reshape(info["shape"]).clone()
    return out


def rel_l2(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.double().flatten(), b.double().flatten()
    return float((a - b).norm() / b.norm().clamp_min(1e-30))


def max_abs(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.double() - b.double()).abs().max())


@torch.inference_mode()
def run_fork(draft, target_embed, target_head, dcache, ref: dict, device) -> dict:
    """The production draft path on one anchor: context K/V from the taps, then the block forward."""
    P = int(ref["anchor_pos"][0])
    anchor = int(ref["anchor_token"][0])
    cfg = draft.config
    taps = [ref[f"tap_{n}"][:P].to(device).half()[None] for n in cfg.target_layer_ids]
    block_table = torch.arange(dcache.max_num_tokens // 256, dtype = torch.int32)[None]
    st = {}

    # Stage fc / hidden_norm: the same two calls the K/V update makes
    cat = torch.cat(taps, dim = -1)
    fc = draft.input_layer.proj.forward(cat, {}, out_dtype = torch.half)
    st["fc"] = fc[0].float().cpu()
    st["hidden_norm"] = draft.input_layer.norm.forward(fc, {}, out_dtype = torch.half)[0].float().cpu()

    # Context K/V written to the paged cache by the production routine
    draft.update_kv_from_target(
        target_hidden = [t.clone() for t in taps],
        cache = dcache,
        params = {"block_table": block_table, "cache_seqlens": torch.tensor([0], dtype = torch.int32)},
    )
    for l, attn in enumerate(draft.attn_modules):
        layer = dcache.layers[(attn.layer_idx, 0)]
        k = layer.k.reshape(-1, attn.num_kv_heads, attn.head_dim)[:P]
        v = layer.v.reshape(-1, attn.num_kv_heads, attn.head_dim)[:P]
        st[f"ctx_k{l}"] = k.float().cpu()
        st[f"ctx_v{l}"] = v.float().cpu()

    # Block forward, module by module (what Model.forward_ls runs)
    params = {
        "attn_mode": "flash_attn",
        "block_table": block_table,
        "cache": dcache,
        "cache_seqlens": torch.tensor([P], dtype = torch.int32),
    }
    ids = torch.tensor([[anchor]], dtype = torch.long)
    x = draft.prepare_inputs(ids, params)
    layer_no = 0
    for module, instance, idx in draft.fwd_modules:
        params["layer_instance"] = instance
        x = module.prepare_for_device(x, params)
        x = module.forward(x, params)
        name = module.module_name if hasattr(module, "module_name") else module.__class__.__name__
        if module is draft.input_layer:
            st["block_embed"] = x.float().cpu()
        elif module.__class__.__name__ == "TransformerBlock":
            st[f"layer{layer_no}_out"] = x.float().cpu()
            layer_no += 1
        elif module.key == "norm":
            st["final_norm"] = x.float().cpu()
            final = x
    st["_final"] = final

    # Base logits through the target's own (quantized) LM head
    lg = target_head.forward(target_head.prepare_for_device(final.half(), params), params)
    st["base_logits_fork"] = lg[0].float().cpu()
    st["_logits_dev"] = lg
    st["_anchor"] = torch.tensor([anchor], dtype = torch.long, device = device)
    return st


def stage_table(fork: dict, ref: dict, floors: dict, P: int, names: list[str]) -> list[dict]:
    rows = []
    for name in names:
        if name not in ref or name not in fork:
            continue
        f, r = fork[name], ref[name]
        if name in ("fc", "hidden_norm") or name.startswith("ctx_"):
            f, r = f[:P], r[:P]
        row = {"stage": name, "numel": int(r.numel()), "rel_l2": rel_l2(f, r), "max_abs": max_abs(f, r),
               "ref_rms": float(r.double().pow(2).mean().sqrt())}
        for fname, fl in floors.items():
            if name in fl:
                fr = fl[name]
                if name in ("fc", "hidden_norm") or name.startswith("ctx_"):
                    fr = fr[:P]
                row[f"floor_{fname}"] = rel_l2(fr, r)
        rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser(description = __doc__.split("\n")[1])
    ap.add_argument("--draft", required = True)
    ap.add_argument("--target", required = True, help = "target pack or checkpoint: only its embedding and LM head are loaded")
    ap.add_argument("--ref", action = "append", required = True, help = "float32 reference dump directory (repeatable)")
    ap.add_argument("--floor", action = "append", default = [], help = "name=dir of lower-precision dumps of the same anchors, "
                    "paired with --ref in the same order per name (repeatable)")
    ap.add_argument("--bf16-head", default = None, help = "checkpoint directory holding the bf16 lm_head.weight: "
                    "the reference's head, to separate the head's own quantization from the draft body")
    ap.add_argument("--factor", type = float, default = 2.0)
    ap.add_argument("--min-tol", type = float, default = 2e-3)
    ap.add_argument("--device", default = "cuda:0")
    ap.add_argument("--out", default = None)
    args = ap.parse_args()
    device = torch.device(args.device)

    floors_by_name = {}
    for spec in args.floor:
        name, path = spec.split("=", 1)
        floors_by_name.setdefault(name, []).append(path)

    dcfg = Config.from_directory(args.draft)
    draft = Model.from_config(dcfg)
    tcfg = Config.from_directory(args.target)
    target = Model.from_config(tcfg)
    draft.attach_to(target)

    dcache = Cache(draft, max_num_tokens = 4096, max_batch_size = 1)
    draft.load(device, progressbar = False)
    embed = target.modules[0]
    head = target.modules[target.logit_layer_idx]
    for m in (embed, head):
        tcfg.stc.begin_deferred_load()
        m.load(torch.device("cpu") if m.caps.get("prefer_cpu") else device)
        tcfg.stc.end_deferred_load()

    head_bf16 = None
    if args.bf16_head:
        from safetensors import safe_open
        idx = json.load(open(os.path.join(args.bf16_head, "model.safetensors.index.json")))["weight_map"]
        with safe_open(os.path.join(args.bf16_head, idx["lm_head.weight"]), framework = "pt", device = "cpu") as f:
            head_bf16 = f.get_tensor("lm_head.weight").to(device)

    k_list = (16, 32, 64)
    report = {"draft": os.path.basename(os.path.normpath(args.draft)), "anchors": [], "pass": True}
    for ai, ref_dir in enumerate(args.ref):
        ref = read_dump(ref_dir)
        P = int(ref["anchor_pos"][0])
        floors = {n: read_dump(paths[ai]) for n, paths in floors_by_name.items()}
        fork = run_fork(draft, embed, head, dcache, ref, device)
        entry = {"ref": os.path.basename(os.path.normpath(ref_dir)), "P": P, "anchor_token": int(ref["anchor_token"][0])}

        # (1) draft body: everything up to the final norm
        body = ["fc", "hidden_norm"] + [f"ctx_{kv}{l}" for l in range(5) for kv in "kv"] + \
               ["block_embed"] + [f"layer{l}_out" for l in range(5)] + ["final_norm"]
        rows = stage_table(fork, ref, floors, P, body)

        # (2) base logits: the fork's quantized head, and the bf16 head applied to the fork's final norm
        lg = fork["base_logits_fork"]
        rows += stage_table({"base_logits": lg}, ref, floors, P, ["base_logits"])
        if head_bf16 is not None:
            hb = torch.nn.functional.linear(fork["_final"].half()[0], head_bf16).float().cpu()
            r = stage_table({"base_logits_bf16head": hb}, {"base_logits_bf16head": ref["base_logits"]},
                            {n: {"base_logits_bf16head": f["base_logits"]} for n, f in floors.items() if "base_logits" in f},
                            P, ["base_logits_bf16head"])
            rows += r
            lg_use = hb.to(device)[None].half()
        else:
            lg_use = fork["_logits_dev"]
        entry["stages"] = rows

        # (3) the walk in isolation: the reference's own base logits in, picks out (exact)
        anchor = fork["_anchor"]
        base = ref["base_logits"].to(device)[None]
        walk = []
        for K in k_list:
            u, c = markov_candidates(base, base.shape[-1], K)
            out = markov_walk(u, c, anchor, draft.heads.w1, draft.heads.w2)[0, 1:].cpu()
            tok_ref = ref[f"tokens_{K}"].long()
            cand_ok = bool((torch.sort(c[0].cpu(), dim = -1).values ==
                            torch.sort(ref[f"idx_{K}"].long(), dim = -1).values).all())
            walk.append({"K": K, "tokens_equal": bool((out == tok_ref).all()), "n_equal": int((out == tok_ref).sum()),
                         "candidates_equal": cand_ok})
            if not walk[-1]["tokens_equal"] or not cand_ok:
                entry["pass_walk_isolated"] = False
        entry.setdefault("pass_walk_isolated", True)
        entry["walk_isolated"] = walk

        # (4) picks and confidence end to end from the fork's own logits
        e2e = []
        for K in k_list:
            u, c = markov_candidates(lg_use, lg_use.shape[-1], K)
            out = markov_walk(u, c, anchor, draft.heads.w1, draft.heads.w2)
            toks = out[0, 1:].cpu()
            tok_ref = ref[f"tokens_{K}"].long()
            eq = (toks == tok_ref)
            lead = int(torch.cumprod(eq.long(), 0).sum())
            conf = markov_confidence(fork["_final"], out[:, :-1], draft.heads.w1, draft.heads.conf_w, draft.heads.conf_b)[0].cpu()
            conf_ref = ref[f"conf_{K}"].float()
            e2e.append({"K": K, "tokens_equal": int(eq.sum()), "of": int(eq.numel()), "leading_equal": lead,
                        "conf_max_abs_vs_ref": max_abs(conf, conf_ref),
                        "conf_max_abs_where_prefix_equal": max_abs(conf[:lead], conf_ref[:lead]) if lead else None})
        entry["picks_end_to_end"] = e2e

        # pass: every body stage within factor x floor (floor = largest of the given lower-precision runs)
        ok = True
        for row in rows:
            fl = [v for k, v in row.items() if k.startswith("floor_")]
            tol = max(args.min_tol, args.factor * max(fl)) if fl else args.min_tol
            row["tol"] = tol
            row["ok"] = row["rel_l2"] <= tol
            # the quantized head is expected to sit above the bf16 head's floor: reported, not gated
            if row["stage"] == "base_logits":
                row["ok"] = True
                row["note"] = "target LM head is quantized: expected above the floor"
            ok = ok and row["ok"]
        entry["pass_stages"] = ok
        report["pass"] = report["pass"] and ok and entry["pass_walk_isolated"]
        report["anchors"].append(entry)

        print(f"anchor {entry['ref']} P={P}")
        print(f"  {'stage':20s} {'rel_l2':>10s} {'max_abs':>10s} " + " ".join(f"{'floor_' + n:>12s}" for n in floors) + f" {'tol':>10s}  ok")
        for row in rows:
            print(f"  {row['stage']:20s} {row['rel_l2']:10.3e} {row['max_abs']:10.3e} " +
                  " ".join(f"{row.get('floor_' + n, float('nan')):12.3e}" for n in floors) + f" {row['tol']:10.3e}  {row['ok']}")
        for w in walk:
            print(f"  walk isolated K={w['K']:3d}: tokens equal {w['n_equal']}/15, candidates equal {w['candidates_equal']}")
        for e in e2e:
            print(f"  picks e2e   K={e['K']:3d}: tokens equal {e['tokens_equal']}/{e['of']} (leading {e['leading_equal']}), "
                  f"conf max_abs {e['conf_max_abs_vs_ref']:.3e}")

    print("GATE", "PASS" if report["pass"] else "FAIL")
    if args.out:
        json.dump(report, open(args.out, "w"), indent = 1)
    sys.exit(0 if report["pass"] else 1)


if __name__ == "__main__":
    main()
