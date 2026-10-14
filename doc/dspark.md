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
   one of the verify pass at every temperature.
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
draft.adaptive_verification = True     # verify as many drafts as pay off (see below); default False

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
| `draft.markov_top_k` | Markov candidates per row. 8, 16 and 32 use the native top-k kernel, other values `torch.topk`. Measured on the reference checkpoint with a torch reference (teacher-forced accepted length over 40 prompts, against the full-vocabulary walk): -3.8% (64), -6% (32), -9% (16) |
| `draft.draft_conf_threshold` | keep only the leading drafts whose confidence is at least this; 0 drafts verified falls back to a plain step. `None` (default) verifies all |
| `draft.adaptive_verification` | per round, verify the number of drafts that maximizes the expected accepted tokens per unit of time under a cost model of a round (needs the confidence head) |
| `draft.set_verify_cost(costs, draft_cost)` | the cost model: the time of one verify pass over 1 .. 16 rows and of the draft side of a round, in any one unit. The default, 1 + 0.1 per extra row, is a guess for a mixture-of-experts target on a bandwidth bound device: `tools/dspark/spec_check.py --tasks profile` measures it |
| `Generator(dynamic_draft_tokens = True)` | the generator's online calibration over the exported confidence instead of a fixed threshold |

Why truncate: verifying a row is not free on a mixture-of-experts target, each extra row activates more experts. On
the measured pair a verify pass costs 16.3 ms for 1 row and 47.3 ms for 16 rows, the draft side 3.7 ms, so a round that
drafts 15 tokens and gets 2 of them accepted (chat prompts) is slower than plain decoding.

## Measured

MiniCPM-V-4.7-35B-A3B, text path, target EXL3 6.0 bpw, DSpark draft EXL3 6.0 bpw (uncalibrated), one GB10, one stream,
greedy, thinking off, at most 320 new tokens, 80 prompts (40 gsm8k, 20 humaneval, 20 ifeval). Tokens per second over
the decode phase of each request; `AL` is accepted length including the bonus token (new tokens per round).

| | plain | 15 drafts, K=64 | adaptive, K=64 | threshold 0.5, K=64 | 15 drafts, K=32 | 15 drafts, K=16 |
| --- | --- | --- | --- | --- | --- | --- |
| gsm8k | 60.5 | 150.9 (AL 8.32) | 173.1 (7.53) | 167.9 (7.93) | 148.2 (8.07) | 141.5 (7.66) |
| code | 60.4 | 199.4 (10.91) | 213.7 (10.33) | 213.6 (10.76) | 193.0 (10.62) | 191.5 (10.35) |
| chat | 60.7 | 42.2 (2.10) | 71.5 (1.93) | 70.0 (1.70) | 41.4 (2.07) | 40.7 (2.02) |

The accepted length is the draft's quality and does not depend on the target's speed. The teacher-forced accepted
length of a torch reference of this draft on the same kind of text is 4.62 over 40 prompts with the full-vocabulary
walk (gsm8k 8.1, code 11.9, chat 2.3) and 4.43 at K = 64; the figures above are what the engine achieves on its own
greedy streams. The speedup depends on what a verify row costs.

## Numerics

- **Draft forward** against a float32 reference built from the same quantized weights: per stage relative L2 error
  6e-4 to 3e-3 up to the final norm, 1.5 to 3.5 times the floor of an fp16 reference run and 3 to 4 times below the
  bf16 floor; the Markov walk on the reference's own logits picks exactly the reference's tokens for K = 16, 32, 64
  (`tools/dspark/stage_gate.py`).
- **Greedy output against plain decoding**: 60 of 80 streams (up to 320 tokens) are identical. The other 20 diverge at
  one token whose top-2 logit gap is at most 0.44 nats, at the same positions for K = 64 and K = 32. The cause is not
  the draft: the single-row decode path and the multi-row paths of this target disagree on the logits by a relative L2
  of about 3e-2 (heavy tailed: mixture-of-experts routing flips), and multi-row windows of different lengths disagree
  with each other by the same amount (windows of 9 to 16 rows are bit-identical to one another), so any verify window
  flips the near ties that single-row decoding resolves the other way. Rewinding is exact: a 16-row window committed
  after k drafts leaves a recurrent state, K/V and draft K/V that are bit-identical to the ones of a window whose
  rejected tail was different (all k, windows of 16, 8, 5 and 2 rows). Against the same tokens run one at a time the
  committed state differs by a relative L2 of up to 3e-2, the same size as the kernel difference above.
- **Truncated windows** use more kernel shapes and so flip a few more ties (23 and 24 of 80 streams for the adaptive
  and the threshold runs, with gaps up to 2 nats at the rarely hit positions).
- **Sampled decoding** is exact by construction (the sample at every verified row is the target's own, the draft
  only decides which rows are looked at). Tests: on a small synthetic target through the real generator
  (`tools/dspark/synthetic_check.py`, 60 000 samples per arm) the first token and the second token given each of the
  three most frequent first tokens follow the target's distributions, both with the DSpark draft's own proposals
  (random weights: every round is rejected) and with proposals replaced by the target's greedy continuation (71% of
  first drafts accepted, partial and full acceptance), p = 0.02 to 0.92 over the seven tests. On the real target, at a
  context with a spread distribution (temperature 2, 5000 samples, one stream): first token chi-square 15.0 on 21
  degrees of freedom (p = 0.82) against the verify pass's own row 0, which was checked to be bit-identical to the
  generator's; plain decoding on the same context 28.8 on 21 (p = 0.12). Smaller runs of 2000 samples gave first-token
  p between 0.003 and 0.58 without a pattern at the larger size: nothing resolvable, below a KL of about 1e-3 nats.

## Limitations

- The target must share the draft's vocabulary size and hidden size; tensor-parallel targets are not supported.
- Proposals are greedy at every temperature. Sampled decoding is exact but, because a draft is accepted only when the
  target's own sample equals it, the accepted length drops as the temperature rises.
- The draft is quantized without calibration (identity Hessian) by `convert.py`; calibrated quantization is not
  implemented.
- Drafting with multimodal inputs has not been tested.
- One verify pass is 16 rows, the limit of the fused recurrent decode path: `num_draft_tokens` is capped at 15.
- Output equality with plain decoding is up to the numerics above: do not expect bit-identical streams from a target
  whose single-row and multi-row kernels differ.

## Checks and measurements

`tools/dspark/stage_gate.py` compares the draft forward, stage by stage, with reference dumps (context projection,
per-layer K/V, block outputs, final norm, base logits, Markov picks, confidence). `tools/dspark/spec_check.py` runs
generator-level checks on a real target (`forced`, `windows`: partial commits and shortened windows against plain
decoding, `rollback`: state comparison after partial commits, `sampled`: distribution tests, `numerics`, `explain`),
the profile of the round and the throughput measurement (`bench`, arms `plain`, `k<K>`, `k<K>c<t>`, `k<K>a`). Both take
the pack directories on the command line; wrap them in whatever GPU arbitration the machine uses. Unit tests that need
no GPU: `tests/test_qwen3_dspark.py`.
