#!/usr/bin/env python3
"""
Verify an EXL3 pack of the MiniCPM-V-4.7 DSpark draft (arch Qwen3DSparkModel) against its bf16 source.

    verify_draft_pack.py --pack PACK --source SRC [--target-dir TARGET_HF] [--device cuda:0] [--json OUT] [--no-gpu]

Checks (exit status 1 if any hard check fails):
  inventory   every source tensor is either carried verbatim (same dtype, shape) or replaced by exactly
              {.trellis I16 [k/16, n/16, 16K], .suh F16 [k], .svh F16 [n], .mul1 I32 []}; nothing else is in the pack;
              no .weight of a quantized linear, nothing under embed_tokens. / lm_head.
  bytes       sha256 of the raw bytes of every verbatim tensor (norms, hidden_norm, markov tables, confidence head)
              equal between source and pack, dtype included.
  rel_l2      ||W_q - W|| / ||W|| per quantized matrix (dequantised with the fork's LinearEXL3.reconstruct_hgemm, x = I),
              max / median per kind.
  forward     the pack loads in the fork (Config.from_directory + Model.load), 5 random taps go through
              update_kv_from_target and one block (anchor + 14 mask rows, real embedding rows of the target when
              --target-dir is given, else random) through forward(); the output state is finite, depends on the taps,
              and is compared (rel_l2, cosine) with the same forward over the bf16 source draft.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import struct
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
FORK = os.path.abspath(os.path.join(HERE, "..", ".."))

QUANT_SUFFIXES = ("trellis", "suh", "svh", "mul1")


# ------------------------------------------------------------------------------------------------ safetensors headers

def read_headers(directory: str) -> dict:
    """name -> dict(dtype, shape, file, beg, end) for every tensor of every *.safetensors in `directory`."""
    out = {}
    for fn in sorted(os.listdir(directory)):
        if not fn.endswith(".safetensors"):
            continue
        path = os.path.join(directory, fn)
        with open(path, "rb") as f:
            (n,) = struct.unpack("<Q", f.read(8))
            h = json.loads(f.read(n))
        base = 8 + n
        for k, v in h.items():
            if k == "__metadata__":
                continue
            out[k] = {
                "dtype": v["dtype"], "shape": list(v["shape"]), "file": path,
                "beg": base + v["data_offsets"][0], "end": base + v["data_offsets"][1],
            }
    return out


def sha256_range(info: dict) -> str:
    h = hashlib.sha256()
    with open(info["file"], "rb") as f:
        f.seek(info["beg"])
        left = info["end"] - info["beg"]
        while left:
            b = f.read(min(left, 1 << 24))
            h.update(b)
            left -= len(b)
    return h.hexdigest()


def numel(shape):
    n = 1
    for d in shape:
        n *= d
    return n


# ------------------------------------------------------------------------------------------------ the recipe

def draft_recipe(cfg: dict):
    """(linears {base: (k, n)}, plain {name: shape}) of the DSpark draft that `cfg` (config.json) describes."""
    h = cfg["hidden_size"]
    layers = cfg["num_hidden_layers"]
    hd = cfg["head_dim"]
    q, kv = cfg["num_attention_heads"] * hd, cfg["num_key_value_heads"] * hd
    inter = cfg["intermediate_size"]
    taps = len(cfg["dflash_config"]["target_layer_ids"])
    rank = cfg["markov_rank"]
    vocab = cfg["vocab_size"]
    dvocab = cfg.get("draft_vocab_size", vocab)
    lin = {"fc": (taps * h, h)}
    plain = {"hidden_norm.weight": [h], "norm.weight": [h]}
    for i in range(layers):
        p = f"layers.{i}"
        plain[f"{p}.input_layernorm.weight"] = [h]
        plain[f"{p}.post_attention_layernorm.weight"] = [h]
        lin[f"{p}.self_attn.q_proj"] = (h, q)
        lin[f"{p}.self_attn.k_proj"] = (h, kv)
        lin[f"{p}.self_attn.v_proj"] = (h, kv)
        lin[f"{p}.self_attn.o_proj"] = (q, h)
        plain[f"{p}.self_attn.q_norm.weight"] = [hd]
        plain[f"{p}.self_attn.k_norm.weight"] = [hd]
        lin[f"{p}.mlp.gate_proj"] = (h, inter)
        lin[f"{p}.mlp.up_proj"] = (h, inter)
        lin[f"{p}.mlp.down_proj"] = (inter, h)
    plain["markov_head.markov_w1.weight"] = [vocab, rank]
    plain["markov_head.markov_w2.weight"] = [dvocab, rank]
    if cfg.get("enable_confidence_head"):
        cin = h + (rank if cfg.get("confidence_head_with_markov") else 0)
        plain["confidence_head.proj.weight"] = [1, cin]
        plain["confidence_head.proj.bias"] = [1]
    return lin, plain


def kind_of(base: str) -> str:
    return "fc" if base == "fc" else base.split(".")[-1]


# ------------------------------------------------------------------------------------------------ checks

def check_inventory(src, pack, lin, plain, report):
    errors = []
    FLOATS = ("BF16", "F16", "F32")
    # source must itself match the recipe (so the recipe is the right yardstick)
    for base, (k, n) in lin.items():
        s = src.get(base + ".weight")
        if s is None or s["shape"] != [n, k] or s["dtype"] not in FLOATS:
            errors.append(f"source {base}.weight: {s and (s['dtype'], s['shape'])}, recipe [{n}, {k}]")
    for name, shape in plain.items():
        s = src.get(name)
        if s is None or s["shape"] != shape:
            errors.append(f"source {name}: {s and (s['dtype'], s['shape'])}, recipe {shape}")
    stray_src = sorted(set(src) - set(plain) - {b + ".weight" for b in lin})
    errors += [f"source tensor outside the recipe: {n}" for n in stray_src]

    accounted = set()
    kbits = {}
    for base, (k, n) in lin.items():
        tr = pack.get(base + ".trellis")
        if tr is None:
            errors.append(f"{base}: no trellis in pack")
            continue
        exp = [k // 16, n // 16]
        if tr["dtype"] != "I16" or tr["shape"][:2] != exp or tr["shape"][2] % 16 or not 1 <= tr["shape"][2] // 16 <= 8:
            errors.append(f"{base}.trellis: {tr['dtype']} {tr['shape']}, need I16 {exp + ['16K']}")
        kbits[base] = tr["shape"][2] // 16
        for suf, dt, shp in (("suh", "F16", [k]), ("svh", "F16", [n]), ("mul1", "I32", [])):
            t = pack.get(f"{base}.{suf}")
            if t is None:
                errors.append(f"{base}.{suf}: missing")
            elif t["dtype"] != dt or t["shape"] != shp:
                errors.append(f"{base}.{suf}: {t['dtype']} {t['shape']}, need {dt} {shp}")
        for suf in QUANT_SUFFIXES:
            accounted.add(f"{base}.{suf}")
        if base + ".weight" in pack:
            errors.append(f"{base}.weight is still in the pack next to the EXL3 tensors")
    for name, shape in plain.items():
        t = pack.get(name)
        s = src.get(name)
        if t is None:
            errors.append(f"{name}: missing from pack (verbatim tensor dropped)")
            continue
        accounted.add(name)
        if t["dtype"] != s["dtype"] or t["shape"] != s["shape"]:
            errors.append(f"{name}: pack {t['dtype']} {t['shape']} vs source {s['dtype']} {s['shape']}")
    strays = sorted(set(pack) - accounted)
    errors += [f"pack tensor outside the recipe: {n}" for n in strays]
    for pre in ("embed_tokens.", "lm_head."):
        errors += [f"{n}: the draft shares the target's embedding/head" for n in pack if n.startswith(pre)]

    report["tensor_counts"] = {"source": len(src), "pack": len(pack), "quantized_linears": len(kbits),
                               "verbatim": len(plain)}
    report["K_by_linear"] = kbits
    # bytes
    body_q = sum(numel(pack[b + ".trellis"]["shape"]) * 2 for b in kbits)
    body_s = sum(numel(pack[b + s]["shape"]) * 2 for b in kbits for s in (".suh", ".svh")) + 4 * len(kbits)
    weights = sum(k * n for k, n in lin.values())
    verb = {n: pack[n]["end"] - pack[n]["beg"] for n in plain if n in pack}
    markov = sum(v for n, v in verb.items() if n.startswith("markov_head."))
    report["bytes"] = {
        "quantized_weights": weights,
        "trellis_bytes": body_q, "scale_bytes": body_s,
        "body_bpw_trellis": 8 * body_q / weights, "body_bpw_with_scales": 8 * (body_q + body_s) / weights,
        "markov_bytes": markov,
        "other_verbatim_bytes": sum(verb.values()) - markov,
        "tensor_bytes_total": body_q + body_s + sum(verb.values()),
    }
    return errors


def check_bytes(src, pack, plain):
    errors, rows = [], {}
    for name in plain:
        if name not in src or name not in pack:
            continue
        a, b = sha256_range(src[name]), sha256_range(pack[name])
        rows[name] = {"sha256": a, "equal": a == b, "dtype": pack[name]["dtype"], "shape": pack[name]["shape"]}
        if a != b:
            errors.append(f"{name}: sha256 differs between source and pack")
    return errors, rows


# ------------------------------------------------------------------------------------------------ GPU checks

def load_fork():
    if FORK not in sys.path:
        sys.path.insert(0, FORK)


def dequantize(parts: dict, device: str):
    import torch
    from exllamav3.modules.quant.exl3 import LinearEXL3
    tr = parts["trellis"].to(device)
    k, n = tr.shape[0] * 16, tr.shape[1] * 16
    kw = {"suh": parts["suh"].to(device), "svh": parts["svh"].to(device), "trellis": tr}
    if "mul1" in parts:
        kw["mul1"] = parts["mul1"].to(device)
    lin = LinearEXL3(None, k, n, **kw)
    eye = torch.eye(k, dtype=torch.half, device=device)
    return lin.reconstruct_hgemm(eye, torch.half).T.contiguous()


def rel_l2_scan(pack_dir, src_dir, lin, kbits, device):
    import torch
    from safetensors import safe_open
    pack_files = [os.path.join(pack_dir, f) for f in os.listdir(pack_dir) if f.endswith(".safetensors")]
    src_files = [os.path.join(src_dir, f) for f in os.listdir(src_dir) if f.endswith(".safetensors")]
    pk = {}
    for p in pack_files:
        with safe_open(p, "pt", device="cpu") as f:
            for k in f.keys():
                pk[k] = p
    sk = {}
    for p in src_files:
        with safe_open(p, "pt", device="cpu") as f:
            for k in f.keys():
                sk[k] = p
    rows = {}
    for base in lin:
        parts = {}
        for s in QUANT_SUFFIXES:
            with safe_open(pk[f"{base}.{s}"], "pt", device="cpu") as f:
                parts[s] = f.get_tensor(f"{base}.{s}")
        with safe_open(sk[base + ".weight"], "pt", device="cpu") as f:
            w = f.get_tensor(base + ".weight")
        wq = dequantize(parts, device)
        w = w.to(device).float()
        wq = wq[: w.shape[0], : w.shape[1]].float()
        rel = ((wq - w).norm() / w.norm().clamp_min(1e-30)).item()
        rows[base] = {"rel_l2": rel, "K": kbits[base], "finite": bool(torch.isfinite(wq).all().item())}
    return rows


def summarize_rel(rows):
    by = {}
    for base, r in rows.items():
        by.setdefault(kind_of(base), []).append(r["rel_l2"])
    out = {}
    for kind, vals in by.items():
        out[kind] = {"n": len(vals), "max": max(vals), "median": statistics.median(vals), "min": min(vals)}
    allv = [r["rel_l2"] for r in rows.values()]
    out["ALL"] = {"n": len(allv), "max": max(allv), "median": statistics.median(allv), "min": min(allv)}
    return out


def forward_outputs(model_dir, target_dir, device):
    """Load `model_dir` in the fork and run one DSpark block over 5 random taps for each of several (seed, std) cases.
    Returns ({case: out_state [1, block, hidden] float cpu}, info)."""
    import torch
    from exllamav3 import Config, Model, Cache
    cfg = Config.from_directory(model_dir)
    assert cfg.architecture == "Qwen3DSparkModel"
    model = Model.from_config(cfg)
    cache = Cache(model, max_num_tokens = 2048)
    model.load(device)
    dev = torch.device(device)
    hidden = cfg.hidden_size
    mask_id = cfg.mask_token_id
    anchor_id = 1234

    if target_dir:
        from safetensors import safe_open
        index = json.load(open(os.path.join(target_dir, "model.safetensors.index.json")))["weight_map"]
        name = "model.language_model.embed_tokens.weight"
        with safe_open(os.path.join(target_dir, index[name]), "pt", device="cpu") as f:
            sl = f.get_slice(name)
            rows = {i: sl[i:i + 1].float() for i in (anchor_id, mask_id)}
        emb_src = "real target embed_tokens rows"
    else:
        g = torch.Generator().manual_seed(7)
        rows = {i: torch.randn(1, hidden, generator = g) * 0.02 for i in (anchor_id, mask_id)}
        emb_src = "random embeddings"

    class Embedding:
        def forward(self, ids, params):
            flat = [rows[int(i)] for i in ids.view(-1).tolist()]
            return torch.cat(flat, dim = 0).view(*ids.shape, hidden).to(dev).half()

    class Target:
        loaded_tp = False
        modules = [Embedding()]
        class config:
            vocab_size = cfg.vocab_size
            hidden_size = cfg.hidden_size
            num_hidden_layers = cfg.config_dict.get("num_target_layers", 40)
    target = Target()
    model.attach_to(target)

    outs = {}
    seq = 37
    pages = 2048 // 256
    for seed, std in ((1, 1.0), (2, 1.0), (1, 25.0)):
        g = torch.Generator(device = "cpu").manual_seed(seed)
        taps = [(torch.randn(1, seq, hidden, generator = g) * std).half().to(dev) for _ in cfg.target_layer_ids]
        block_table = torch.arange(pages, dtype = torch.int32, device = dev).view(1, pages)
        cache_seqlens = torch.zeros(1, dtype = torch.int32, device = dev)
        model.update_kv_from_target(taps, cache, {"block_table": block_table, "cache_seqlens": cache_seqlens})
        params = {
            "attn_mode": "flash_attn",
            "block_table": block_table,
            "cache": cache,
            "cache_seqlens": torch.tensor([seq], dtype = torch.int32, device = dev),
        }
        out = model.forward(torch.tensor([[anchor_id]], dtype = torch.long), params)
        outs[(seed, std)] = out.float().cpu()
    refused = False
    try:
        model.sample_from_state(out, {})
    except NotImplementedError:
        refused = True
    info = {"embeddings": emb_src, "sample_from_state_refuses": refused, "block_rows": cfg.block_size,
            "tap_layers": cfg.target_layer_ids}
    model.unload()
    del model, cache
    torch.cuda.empty_cache()
    return outs, info


def forward_check(pack_dir, source_dir, target_dir, device, report):
    import torch
    outs, info = forward_outputs(pack_dir, target_dir, device)
    runs = []
    for (seed, std), o in outs.items():
        runs.append({
            "seed": seed, "tap_std": std, "out_shape": list(o.shape), "finite": bool(torch.isfinite(o).all().item()),
            "abs_max": o.abs().max().item(), "rms": o.pow(2).mean().sqrt().item(),
        })
    errors = []
    if not all(r["finite"] and r["out_shape"][1] == info["block_rows"] for r in runs):
        errors.append("forward: non-finite output or wrong block shape")
    # the context taps must change the block state (K/V of the draft cache are really used)
    d = (outs[(1, 1.0)] - outs[(2, 1.0)]).abs().max().item()
    info["taps_change_output_absmax_delta"] = d
    if not d > 0:
        errors.append("forward: output does not depend on the taps")
    # fidelity of the whole 5-layer block against the bf16 draft (fork fp16 path) on the same inputs
    if source_dir:
        ref, _ = forward_outputs(source_dir, target_dir, device)
        cmp = {}
        for key, o in outs.items():
            r = ref[key]
            rel = ((o - r).norm() / r.norm()).item()
            cos = torch.nn.functional.cosine_similarity(o.view(1, -1), r.view(1, -1)).item()
            cmp[f"seed{key[0]}_std{key[1]:g}"] = {"rel_l2": rel, "cosine": cos}
        info["vs_bf16_source"] = cmp
    info["runs"] = runs
    report["forward"] = info
    return errors


# ------------------------------------------------------------------------------------------------ main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", required=True)
    ap.add_argument("--source", required=True)
    ap.add_argument("--target-dir", default=None, help="HF dir of the MiniCPM-V-4.7 target (real embedding rows for the forward)")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--json", default=None)
    ap.add_argument("--no-gpu", action = "store_true")
    args = ap.parse_args()
    load_fork()

    cfg = json.load(open(os.path.join(args.pack, "config.json")))
    src = read_headers(args.source)
    pack = read_headers(args.pack)
    lin, plain = draft_recipe(cfg)
    report = {"pack": args.pack, "source": args.source, "errors": {}}
    errs = report["errors"]

    errs["inventory"] = check_inventory(src, pack, lin, plain, report)
    errs["bytes"], report["verbatim_sha256"] = check_bytes(src, pack, plain)
    mul1_ok = all(f"{b}.mul1" in pack for b in lin)
    report["mul1_marker_on_all_linears"] = mul1_ok
    if not mul1_ok:
        errs["bytes"].append("a linear has no .mul1 marker")

    if not args.no_gpu:
        rows = rel_l2_scan(args.pack, args.source, lin, report["K_by_linear"], args.device)
        report["rel_l2"] = rows
        report["rel_l2_summary"] = summarize_rel(rows)
        errs["rel_l2"] = [f"{b}: non-finite" for b, r in rows.items() if not r["finite"]]
        errs["forward"] = forward_check(args.pack, args.source, args.target_dir, args.device, report)

    ok = not any(errs.values())
    report["ok"] = ok
    if args.json:
        with open(args.json, "w") as f:
            json.dump(report, f, indent = 1)

    print(f"pack     {args.pack}")
    print(f"tensors  {report['tensor_counts']}")
    b = report["bytes"]
    print(f"K        {sorted(set(report['K_by_linear'].values()))}  body bpw (trellis) {b['body_bpw_trellis']:.4f}"
          f"  with scales {b['body_bpw_with_scales']:.4f}")
    print(f"bytes    trellis {b['trellis_bytes']:,}  scales {b['scale_bytes']:,}  markov {b['markov_bytes']:,}"
          f"  other verbatim {b['other_verbatim_bytes']:,}  tensor total {b['tensor_bytes_total']:,}")
    print(f"verbatim {sum(1 for r in report['verbatim_sha256'].values() if r['equal'])}/{len(report['verbatim_sha256'])} sha256-equal")
    if "rel_l2_summary" in report:
        print("rel_l2   kind      n      max   median      min")
        for kind, s in report["rel_l2_summary"].items():
            print(f"         {kind:8s} {s['n']:3d}  {s['max']:.5f}  {s['median']:.5f}  {s['min']:.5f}")
    if "forward" in report:
        for r in report["forward"]["runs"]:
            print(f"forward  seed {r['seed']} tap_std {r['tap_std']:5.1f} shape {r['out_shape']} finite {r['finite']} "
                  f"rms {r['rms']:.3f} absmax {r['abs_max']:.1f} ({report['forward']['embeddings']})")
        for k, v in report["forward"].get("vs_bf16_source", {}).items():
            print(f"forward  vs bf16 draft {k}: rel_l2 {v['rel_l2']:.5f} cosine {v['cosine']:.6f}")
        print(f"forward  sample_from_state refuses: {report['forward']['sample_from_state_refuses']}; "
              f"taps change output (max abs delta {report['forward']['taps_change_output_absmax_delta']:.3f})")
    for section, es in errs.items():
        for e in es[:20]:
            print(f"FAIL [{section}] {e}")
    print("RESULT   " + ("PASS" if ok else "FAIL"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
