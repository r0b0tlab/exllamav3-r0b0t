# r0b0tlab/exllamav3-r0b0t — community fork

An MIT-licensed community fork of [turboderp-org/exllamav3](https://github.com/turboderp-org/exllamav3),
maintained for EXL3 **serving performance** work that is too invasive or too specific for upstream.
Upstream remains the source of truth for correctness; this fork is the place where measured
speedups get built and proven before anything is proposed upstream or folded into a standalone
engine.

## Branch layout

| branch | role |
| --- | --- |
| `community` | **the fork.** Tracking branch for everything we maintain. |
| `master` | clean mirror of upstream `master`. Never commit here; it is the sync point. |
| `minicpm-v4.7` | MiniCPM-V-4.7 text-path loader for plain decoding, on ExLlamaV3 v1.5.0. Commit `4d083d25c1a3`; the plain MiniCPM-V-4.7 EXL3 pack is measured and documented on this commit. |
| `minicpm-v4.7-dspark` | the same loader plus DSpark block-drafter support, for drafted decoding. Commit `5ddae7d12448`, pinned by the drafted profile of the serving repository (v2.1.0 and later). |
| `gb10` | single-node GB10 run support (optional remote import; runs without trellis files). Commit `44636f06`. |
| `dflash2-pathway`, `dflash2-pathway-backup` | superseded. Upstream `dev` landed its own native DFlash2 (CUDA kernels in `exllamav3_ext/dflash2.cu`, `architecture/dflash2.py`, `tests/test_dflash2.py`). Our PR was declined on the grounds that upstream's implementation is better, which it is. Do not build on these. |

`community` is based on upstream **`dev`**, not `master`, because upstream's live work (DFlash2,
GDN rewind, MoE CPU offload) is on `dev`. Rebasing across a `dev` rewrite is expected and accepted;
keep our changes in separate commits with clear messages so a rebase is mechanical.

### Syncing with upstream

```
git fetch origin
git rebase origin/dev        # on `community`
```

Keep our diffs additive and confined to `exllamav3/exllamav3_ext/quant/` where possible. Any
change that touches `exllamav3/modules/` or the generator will conflict often.

## What this fork is for

A measured set of EXL3 serving wins on consumer NVIDIA hardware, built in this order:

1. **Dense M=32/64 tile that shares a decoded B fragment across row-tiles. SHIPPED, measured
   1.35x-1.92x on the verify pass.** On an RTX 3090 with `qwen38-27b-exl3` at 4.000 bpw,
   default (autotuned) path:
   | m | up_proj (5120x17408) | down_proj (17408x5120) |
   | --- | --- | --- |
   | 1 / 8 / 16 | unchanged | unchanged |
   | 32 | 0.120 ms vs 0.162 (1.35x) | 0.121 vs 0.191 (1.58x) |
   | 64 | 0.215 vs 0.319 (1.48x) | 0.202 vs 0.383 (1.87x) |
   | 128 | 0.416 vs 0.619 (1.49x) | 0.410 vs 0.783 (1.92x) |
   | 192 | 0.621 vs 0.927 (1.50x) | 0.609 vs 1.160 (1.92x) |

   Numerics: `TILEBLOCKS_M` contributes **zero** divergence. Shapes 5 and 6 reproduce the
   existing shapes' divergence statistics bit-for-bit against shape 4 — 5-vs-4 matches 3-vs-4
   exactly, 6-vs-4 matches 2-vs-4 exactly — so all of the observed spread is the pre-existing
   (k,n) tiling difference, already present between upstream's own shapes 2, 3 and 4. Relative RMS
   stays at 2.0e-3 .. 3.8e-3 for every pair. Do not "fix" this by tightening tolerances.
2. Per-architecture fixes the upstream tree hardcodes: `SMEM_MAX` is pinned to 90 KB with the
   comment "max shared memory on compute capability 8.6" and is passed straight to
   `cudaFuncSetAttribute` and the cooperative autotuner; the fp16-accumulator MMA gate
   (`EXL3_GEMM_H_ACC`) is `__CUDA_ARCH__ == 860` even though upstream's own measurements record
   2.00x on 3090, 4090 **and 5090**.
3. Whatever the measurements say next. No change lands without a number attached.

## Why: the measurements

Full detail in `notes/E1-ROOFLINE-RESULTS.md` in the sibling working tree. Short version, all
measured on an RTX 3090 against `qwen38-27b-exl3` at 4.000 bpw, mul1 codebook:

- **Speculation saturates at a draft window of 15.** Per-row cost falls 13-14x from m=1 to m=16 and
  then is completely flat out to m=144. Cause: the dense `TILESIZE_M=16` GEMM re-streams the whole
  trellis once per 16-row row-tile. Wall time is exactly proportional to `ceil(m/16)`.
- **A 2.41x fix exists and is prototyped.** Holding compute identical and varying only how often a
  16x16 B tile is decoded: restream 9.216 ms vs shared 3.817 ms at NROW=4.
- **Every cheap lever is exhausted.** No flag, kernel shape, grid size, reconstruct path, or config
  variant beats the shipped default (best forced gain 1.044x). The cooperative grid is capped at
  one block per SM — `num_sms` 164 is a fatal `too many blocks in cooperative launch` assert.
- The trellis decode is not the wall: a standalone read+decode probe sustains 777 GB/s (87% of the
  measured 893 GB/s streaming ceiling) and the achievable ceiling is at least 932 GB/s.
- `exl3_gemm_kernel_inner` **already** implements `TILEBLOCKS_M > 1` with row fragments sharing one
  dequantized B fragment. It is instantiated only for the fused-MoE path. The flagship change is
  exposing that path for dense shapes, not writing a new kernel.

## Building

The extension is a single ~1.7 GB `.so` and a full build is slow. Practical notes:

- CUDA and torch must agree. The working environment has system CUDA 13.0 and `torch 2.10.0+cu130`;
  set `CUDAHOSTCXX` to a g++ wrapper if the host toolchain hits the glibc 2.43 `rsqrt` noexcept
  conflict.
- A wheel built against CUDA 12 needs `libcudart.so.12` on `LD_LIBRARY_PATH` at import time even
  when torch is a cu13 build.

## License

MIT, inherited. `LICENSE` is unchanged and upstream copyright is retained.
