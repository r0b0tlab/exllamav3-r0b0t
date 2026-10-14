#!/usr/bin/env python3
"""
Distribution and losslessness checks of DSpark drafting on a small synthetic target, through the real Generator.

A tiny dense target (4 layers, hidden 256, vocabulary 640, random fp16 weights) and a DSpark-shaped draft (random fp16
weights) are written to temporary directories and loaded with the fork's own modules and kernels. On such a model the
numerical difference between execution shapes is orders of magnitude below what a statistical test over thousands of
samples resolves, so the generator's acceptance and rollback logic is tested on its own:

  sampled   n jobs sample two tokens from one context at temperature T with the draft active. The first token, and the
            second token given the most frequent first tokens, are tested against the target's exact distributions
            (chi-square on pooled cells). Run twice: with the DSpark draft as it is (random weights: its proposals are
            almost never right, every round exercises the rejection path), and with proposals replaced by the target's
            own greedy continuation (frequent partial and full acceptance).
  greedy    drafted greedy decoding of several prompts against plain decoding, both ways.

    python tools/dspark/synthetic_check.py [--samples 12000] [--temperature 1.0]

Needs one GPU; the models are a few megabytes.
"""
import argparse
import json
import math
import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

import torch
from safetensors.torch import save_file

from exllamav3 import Cache, Config, Generator, Job, Model, Tokenizer
from exllamav3.generator.sampler import ArgmaxSampler, CategoricalSampler

V, H, INTER, LAYERS, NQ, NKV, HD = 640, 256, 512, 4, 4, 2, 64          # target
DH, DINTER, DLAYERS, DNQ, DNKV, DHD, RANK = 256, 256, 2, 2, 1, 128, 32  # draft
TAPS = [1, 3]
MASK = 600


def build_target(path: str, g: torch.Generator):
    cfg = {
        "architectures": ["Qwen3ForCausalLM"], "model_type": "qwen3", "hidden_size": H, "intermediate_size": INTER,
        "num_hidden_layers": LAYERS, "num_attention_heads": NQ, "num_key_value_heads": NKV, "head_dim": HD,
        "vocab_size": V, "rms_norm_eps": 1e-6, "rope_theta": 1000000.0, "hidden_act": "silu",
        "max_position_embeddings": 4096, "tie_word_embeddings": False, "bos_token_id": 1, "eos_token_id": 2,
        "torch_dtype": "float16",
    }

    def w(*shape, std):
        return (torch.randn(*shape, generator = g) * std).half()

    t = {"model.embed_tokens.weight": w(V, H, std = 1.0), "model.norm.weight": torch.ones(H).half(),
         "lm_head.weight": w(V, H, std = 0.42)}
    for i in range(LAYERS):
        p = f"model.layers.{i}."
        t[p + "input_layernorm.weight"] = torch.ones(H).half()
        t[p + "post_attention_layernorm.weight"] = torch.ones(H).half()
        t[p + "self_attn.q_norm.weight"] = torch.ones(HD).half()
        t[p + "self_attn.k_norm.weight"] = torch.ones(HD).half()
        t[p + "self_attn.q_proj.weight"] = w(NQ * HD, H, std = 0.08)
        t[p + "self_attn.k_proj.weight"] = w(NKV * HD, H, std = 0.08)
        t[p + "self_attn.v_proj.weight"] = w(NKV * HD, H, std = 0.08)
        t[p + "self_attn.o_proj.weight"] = w(H, NQ * HD, std = 0.08)
        t[p + "mlp.gate_proj.weight"] = w(INTER, H, std = 0.06)
        t[p + "mlp.up_proj.weight"] = w(INTER, H, std = 0.06)
        t[p + "mlp.down_proj.weight"] = w(H, INTER, std = 0.06)
    os.makedirs(path, exist_ok = True)
    save_file(t, os.path.join(path, "model.safetensors"))
    json.dump(cfg, open(os.path.join(path, "config.json"), "w"))

    # A word-level tokenizer: "t0" ... "t639"
    from tokenizers import Tokenizer as HFTokenizer, models, pre_tokenizers
    tk = HFTokenizer(models.WordLevel({f"t{i}": i for i in range(V)}, unk_token = "t0"))
    tk.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tk.save(os.path.join(path, "tokenizer.json"))


def build_draft(path: str, g: torch.Generator):
    cfg = {
        "architectures": ["Qwen3DSparkModel"], "attention_bias": False, "block_size": 15, "bos_token_id": 1,
        "dflash_config": {
            "attention_mode": "gqa", "confidence_head_alpha": 1.0, "confidence_head_with_markov": True,
            "enable_confidence_head": True, "markov_head_type": "vanilla", "markov_rank": RANK, "mask_token_id": MASK,
            "projector_type": "dspark", "target_layer_ids": TAPS,
        },
        "draft_vocab_size": V, "dtype": "float16", "eos_token_id": 2, "head_dim": DHD, "hidden_act": "silu",
        "hidden_size": DH, "intermediate_size": DINTER, "layer_types": ["full_attention"] * DLAYERS,
        "max_position_embeddings": 4096, "model_type": "qwen3", "num_attention_heads": DNQ,
        "num_hidden_layers": DLAYERS, "num_key_value_heads": DNKV, "num_target_layers": LAYERS, "pad_token_id": 0,
        "rms_norm_eps": 1e-6, "rope_parameters": {"rope_theta": 10000000, "rope_type": "default", "rope_dim": DHD},
        "sliding_window": None, "use_sliding_window": False, "vocab_size": V, "markov_rank": RANK,
        "enable_confidence_head": True, "confidence_head_with_markov": True,
    }

    def w(*shape, std):
        return (torch.randn(*shape, generator = g) * std).half()

    t = {
        "fc.weight": w(DH, DH * len(TAPS), std = 0.05), "hidden_norm.weight": torch.ones(DH).half(),
        "norm.weight": torch.ones(DH).half(),
        "markov_head.markov_w1.weight": w(V, RANK, std = 0.5), "markov_head.markov_w2.weight": w(V, RANK, std = 0.5),
        "confidence_head.proj.weight": w(1, DH + RANK, std = 0.05), "confidence_head.proj.bias": torch.zeros(1).half(),
    }
    for i in range(DLAYERS):
        p = f"layers.{i}."
        t[p + "input_layernorm.weight"] = torch.ones(DH).half()
        t[p + "post_attention_layernorm.weight"] = torch.ones(DH).half()
        t[p + "self_attn.q_norm.weight"] = torch.ones(DHD).half()
        t[p + "self_attn.k_norm.weight"] = torch.ones(DHD).half()
        t[p + "self_attn.q_proj.weight"] = w(DNQ * DHD, DH, std = 0.06)
        t[p + "self_attn.k_proj.weight"] = w(DNKV * DHD, DH, std = 0.06)
        t[p + "self_attn.v_proj.weight"] = w(DNKV * DHD, DH, std = 0.06)
        t[p + "self_attn.o_proj.weight"] = w(DH, DNQ * DHD, std = 0.06)
        t[p + "mlp.gate_proj.weight"] = w(DINTER, DH, std = 0.06)
        t[p + "mlp.up_proj.weight"] = w(DINTER, DH, std = 0.06)
        t[p + "mlp.down_proj.weight"] = w(DH, DINTER, std = 0.06)
    os.makedirs(path, exist_ok = True)
    save_file(t, os.path.join(path, "model.safetensors"))
    json.dump(cfg, open(os.path.join(path, "config.json"), "w"))


def chi_square(counts: torch.Tensor, probs: torch.Tensor, min_expected: float = 5.0):
    n = float(counts.sum())
    exp = probs.double() * n
    order = torch.argsort(exp, descending = True)
    big = order[exp[order] >= min_expected]
    small = order[exp[order] < min_expected]
    o = counts[big].double().tolist()
    e = exp[big].tolist()
    if len(small):
        o.append(float(counts[small].sum()))
        e.append(float(exp[small].sum()))
    stat = sum((a - b) ** 2 / b for a, b in zip(o, e) if b > 0)
    dof = max(1, len(o) - 1)
    z = ((stat / dof) ** (1 / 3) - (1 - 2 / (9 * dof))) / math.sqrt(2 / (9 * dof))
    return stat, dof, 0.5 * math.erfc(z / math.sqrt(2))


def run(gen, ids, n_new, sampler, ident = 0, **kw):
    job = Job(input_ids = torch.tensor([ids], dtype = torch.long), max_new_tokens = n_new, sampler = sampler,
              identifier = ident, stop_conditions = [], decode_special_tokens = True, **kw)
    gen.enqueue(job)
    toks, logits = [], None
    while gen.num_remaining_jobs():
        for r in gen.iterate():
            if r.get("identifier") != ident:
                continue
            for src in (r, r.get("held") or {}):
                if "token_ids" in src:
                    toks.extend(int(x) for x in src["token_ids"].flatten().tolist())
                if "logits" in src and logits is None:
                    logits = src["logits"].flatten(0, -2)[0].float().cpu()
    return toks, logits, job


def main():
    ap = argparse.ArgumentParser(description = __doc__.split("\n")[1])
    ap.add_argument("--samples", type = int, default = 12000)
    ap.add_argument("--temperature", type = float, default = 1.0)
    ap.add_argument("--seed", type = int, default = 0)
    ap.add_argument("--device", default = "cuda:0")
    args = ap.parse_args()
    T, n = args.temperature, args.samples
    g = torch.Generator().manual_seed(args.seed)

    tmp = tempfile.TemporaryDirectory()
    tdir, ddir = os.path.join(tmp.name, "target"), os.path.join(tmp.name, "draft")
    build_target(tdir, g)
    build_draft(ddir, g)

    tcfg = Config.from_directory(tdir)
    target = Model.from_config(tcfg)
    draft = Model.from_config(Config.from_directory(ddir))
    cache_p = Cache(target, max_num_tokens = 4096, max_batch_size = 4)
    cache_s = Cache(target, max_num_tokens = 4096, max_batch_size = 4, max_history = 15)
    dcache = Cache(draft, max_num_tokens = 4096)
    target.load(args.device, progressbar = False)
    draft.load(args.device, progressbar = False)
    tok = Tokenizer.from_config(tcfg)
    gen_plain = Generator(model = target, cache = cache_p, tokenizer = tok, max_batch_size = 4)
    gen_spec = Generator(model = target, cache = cache_s, tokenizer = tok, max_batch_size = 4, draft_model = draft,
                         draft_cache = dcache, num_draft_tokens = 15, record_draft_stats = True)
    orig = draft.sample_from_state
    result = {"temperature": T, "samples": n}

    # --- exact distributions of the target
    ctx = torch.randint(3, MASK, (24,), generator = g).tolist()
    _, lg0, _ = run(gen_plain, ctx, 1, ArgmaxSampler(), return_logits = True)
    p0 = torch.softmax(lg0 / T, dim = -1)
    result["context_entropy_nats"] = float(-(p0 * p0.clamp_min(1e-30).log()).sum())
    result["p_top"] = float(p0.max())

    greedy_cache = {}
    truth = {"ids": None}

    def oracle(state, params):
        """proposals = the target's own greedy continuation: of the stream truth["ids"] when it is set (a plain run of
        the same prompt), else of the prefix ctx[:P] + [anchor] (a plain generator run, cached)"""
        out = orig(state, params)
        rows = []
        for b in range(out.shape[0]):
            P = int(params["cache_seqlens"][b])
            anchor = int(out[b, 0])
            if truth["ids"] is not None:
                nxt = truth["ids"][P + 1:P + 16]
                nxt = nxt + [(anchor + 1) % V] * (15 - len(nxt))
            else:
                prefix = tuple(ctx[:P]) + (anchor,)
                if prefix not in greedy_cache:
                    greedy_cache[prefix], _, _ = run(gen_plain, list(prefix), 15, ArgmaxSampler(), ident = 7)
                nxt = greedy_cache[prefix]
            rows.append([anchor] + nxt)
        return torch.tensor(rows, dtype = torch.long, device = out.device)

    def sample_two(label):
        counts1 = torch.zeros(V, dtype = torch.long)
        pair = {}
        t0 = time.time()
        for i in range(n):
            gen_spec.enqueue(Job(
                input_ids = torch.tensor([ctx], dtype = torch.long), max_new_tokens = 2,
                sampler = CategoricalSampler(temperature = T), identifier = i, stop_conditions = [],
                decode_special_tokens = True,
            ))
        toks = {i: [] for i in range(n)}
        acc_of = {}
        while gen_spec.num_remaining_jobs():
            for r in gen_spec.iterate():
                i = r.get("identifier")
                if i is None:
                    continue
                if r.get("eos"):
                    acc_of[i] = r.get("accepted_draft_tokens", 0)
                for src in (r, r.get("held") or {}):
                    if "token_ids" in src:
                        toks[i].extend(int(x) for x in src["token_ids"].flatten().tolist())
        accepted = 0
        for i in range(n):
            a, b = toks[i][:2]
            counts1[a] += 1
            pair.setdefault(a, []).append(b)
            accepted += acc_of.get(i, 0)
        stat, dof, p = chi_square(counts1, p0)
        entry = {"proposals": label, "secs": round(time.time() - t0, 1), "drafts_accepted_in_first_round": accepted,
                 "first_token": {"chi2": stat, "dof": dof, "p": p}, "second_given_first": []}
        for c in torch.topk(counts1, 4).indices.tolist():
            sec = pair.get(c, [])
            if len(sec) < 200:
                continue
            _, lg1, _ = run(gen_plain, ctx + [c], 1, ArgmaxSampler(), return_logits = True)
            cnt = torch.zeros(V, dtype = torch.long)
            for b in sec:
                cnt[b] += 1
            stat, dof, p = chi_square(cnt, torch.softmax(lg1 / T, dim = -1))
            entry["second_given_first"].append({"first": c, "n": len(sec), "chi2": stat, "dof": dof, "p": p})
        print(f"[sampled] {json.dumps(entry)}", flush = True)
        return entry

    result["sampled"] = []
    draft.markov_top_k = 64
    draft.sample_from_state = orig
    result["sampled"].append(sample_two("dspark draft (random weights)"))
    draft.sample_from_state = oracle
    result["sampled"].append(sample_two("target's greedy continuation"))
    draft.sample_from_state = orig

    # --- greedy streams against plain decoding. A stream may differ only where the target itself is torn between two
    # tokens (single-row and multi-row kernels round differently, the logits are fp16): report the gap there
    prompts = [torch.randint(3, MASK, (int(20 + 10 * (k % 5)),), generator = g).tolist() for k in range(12)]
    same = {}
    diffs = []
    for label, fn in (("dspark draft (random weights)", orig), ("target's greedy continuation", oracle)):
        draft.sample_from_state = fn
        same[label] = 0
        for ids in prompts:
            a, _, _ = run(gen_plain, ids, 96, ArgmaxSampler())
            truth["ids"] = ids + a
            b, _, job = run(gen_spec, ids, 96, ArgmaxSampler())
            truth["ids"] = None
            if a == b:
                same[label] += 1
                continue
            j = next(i for i in range(min(len(a), len(b))) if a[i] != b[i])
            _, lg, _ = run(gen_plain, ids + a[:j], 1, ArgmaxSampler(), return_logits = True)
            diffs.append({"proposals": label, "position": j, "logit_gap_nats": float(lg[a[j]] - lg[b[j]]),
                          "top2_gap_nats": float(torch.topk(lg, 2).values.diff().abs())})
    draft.sample_from_state = orig
    result["greedy_identical_streams"] = {k: f"{v}/{len(prompts)}" for k, v in same.items()}
    result["greedy_differences"] = diffs
    print(f"[greedy] identical streams: {json.dumps(result['greedy_identical_streams'])}; differences: {json.dumps(diffs)}",
          flush = True)
    ok = all(e["first_token"]["p"] > 1e-3 and all(s_["p"] > 1e-3 for s_ in e["second_given_first"])
             for e in result["sampled"])
    ok = ok and all(abs(d["logit_gap_nats"]) < 0.05 for d in diffs)
    print("SYNTHETIC CHECK", "PASS" if ok else "FAIL")
    tmp.cleanup()
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
