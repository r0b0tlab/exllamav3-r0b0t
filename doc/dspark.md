# DSpark speculative decoding

`Qwen3DSparkModel` drafts a whole block of tokens in one parallel pass (DFlash style: the target's hidden states are
the context, the block is a fixed-size non-causal query) and then injects the dependency between neighbouring drafts
with a small sequential Markov head. The target verifies the block in one forward pass and keeps the longest prefix
that matches, so the output is the target's own, for greedy and for sampled decoding.

The semantics follow the reference implementation (vLLM `qwen3_dspark.py` and its speculator, Apache-2.0). The
reference checkpoint is `openbmb/MiniCPM-V-4.7-35B-A3B-DSpark`: 5 full-attention layers, 8 query / 2 key-value heads of
width 256, hidden size 2048, block size 15, Markov rank 256, a confidence head, no embedding and no LM head of its own
(the target's are used).

## What a round does

1. The target's residual stream after decoder layers `target_layer_ids` (3, 11, 19, 27, 35 for the reference
   checkpoint) is exported for every verified token and projected, `fc` then `hidden_norm`, into one context row per
   position. Every draft layer turns the row into K/V directly (no input norm, K normed per head and rotated at the
   row's own position, V raw); the K/V pages live in the draft cache, aligned with the target's block tables.
2. The block is `[anchor, mask x 14]` at positions `P .. P + 14`, where `P` is the number of committed positions and
   the anchor is the last sampled token (not yet processed by the target). All block rows see the whole context and the
   whole block. The final-normed block rows go through the **target's** LM head.
3. Row `k` predicts the token after row `k` (the anchor row predicts the first draft), so a round proposes 15 tokens
   and the target verifies 16 rows (anchor + drafts).
4. The Markov bias `markov_w1[x_{k-1}] @ markov_w2^T` is added to row `k`'s logits, with `x_{-1}` the anchor and
   `x_{k-1}` the draft chosen for row `k - 1`. It is evaluated for the **top-K candidates of the base logits** of each
   row only, by the DFlash2 selector-walk kernels (`dflash2_topk`, `dflash2_selector_walk`: predecessor table
   `markov_w1`, successor table `markov_w2`, unary term the base candidate logits, an all-ones gate). Proposals are the
   greedy walk.
5. Verification is the generator's existing accept-while-match: the target samples every verified row with the job's
   sampler and a draft is kept only if it equals that sample. With greedy proposals the proposal distribution is a
   point mass, and this rule is exactly rejection sampling with such a proposal: the output distribution equals the
   target's at every temperature (`tools/dspark/spec_check.py --tasks sampled` tests it).
6. Rejections roll back the target's recurrent (GatedDeltaNet) state to the committed position with the per-token
   history kept during the verify pass, and the paged K/V by position; the draft's context rows past the committed
   length are overwritten by the next round.

The optional confidence head, `sigmoid(proj([h_k ; markov_w1[x_{k-1}]]))`, estimates the probability that draft `k` is
accepted given that the earlier drafts were. It can shorten the verified prefix; this never changes the output.

## Use

```python
from exllamav3 import Cache, Config, Generator, Job, Model, Tokenizer

config = Config.from_directory("target")
model = Model.from_config(config)
# Hybrid (recurrent) targets keep one state snapshot per verified row: max_history must cover the drafts
cache = Cache(model, max_num_tokens = 8192, max_batch_size = 4, max_history = 15)

draft_config = Config.from_directory("dspark-draft")
draft = Model.from_config(draft_config)
draft_cache = Cache(draft, max_num_tokens = 8192)       # same size as the target cache

model.load()
draft.load()

draft.markov_top_k = 64                # candidates per row for the Markov walk (default 64)
draft.draft_conf_threshold = None      # or e.g. 0.5: verify only the leading drafts with confidence >= 0.5

generator = Generator(
    model, cache, Tokenizer.from_config(config),
    draft_model = draft, draft_cache = draft_cache,
    num_draft_tokens = 15,             # at most the block size; a smaller value shortens the verified window
)
```

| setting | meaning |
| --- | --- |
| `Cache(..., max_history = 15)` | recurrent targets only: snapshots kept per verify pass (>= `num_draft_tokens`) |
| `num_draft_tokens` | drafts verified per round, 1 to 15 (the block always runs at its full size) |
| `draft.markov_top_k` | Markov candidates per row. 8, 16 and 32 use the native top-k kernel, other values `torch.topk`. The reference measured the accepted length at -3.8% (64), -6% (32), -9% (16) against the full-vocabulary walk |
| `draft.draft_conf_threshold` | keep only the leading drafts whose confidence is at least this; 0 drafts verified falls back to a plain step. `None` (default) verifies all |
| `Generator(dynamic_draft_tokens = True)` | the generator's online calibration over the exported confidence instead of a fixed threshold |

## Limitations

- The target must share the draft's vocabulary size and hidden size; tensor-parallel targets are not supported.
- Proposals are greedy at every temperature. Sampled decoding is exact but, because a draft is accepted only when the
  target's own sample equals it, the accepted length drops as the temperature rises.
- The draft is quantized without calibration (identity Hessian) by `convert.py`; calibrated quantization is not
  implemented.
- Multimodal inputs are not supported while drafting (as for the other draft models).
- One verify pass is 16 rows, the limit of the fused recurrent decode path: `num_draft_tokens` is capped at 15.

## Checks and measurements

`tools/dspark/stage_gate.py` compares the draft forward, stage by stage, with reference dumps (context projection,
per-layer K/V, block outputs, final norm, base logits, Markov picks, confidence) and `tools/dspark/spec_check.py`
runs the generator-level checks (losslessness against plain decoding, forced partial commits, rollback exactness,
sampled-decoding statistics) and the throughput measurement. Both take the pack directories on the command line and
have to be wrapped in whatever GPU arbitration the machine uses. Unit tests that need no GPU:
`tests/test_qwen3_dspark.py`.
