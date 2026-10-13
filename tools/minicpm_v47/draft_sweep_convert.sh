#!/usr/bin/env bash
# Convert the MiniCPM-V-4.7 DSpark draft (arch Qwen3DSparkModel) to EXL3 at one or more bitrates, one pack per bitrate,
# each with a CONVERT.txt (exact command, fork commit, library patch hash, wall-clock). Runs the stock convert.py of the
# fork checkout this script lives in; the draft is tiny (389M params), a whole sweep point takes minutes.
#
#   tools/minicpm_v47/draft_sweep_convert.sh 6.00 8.00 5.00 4.00 3.00
#   tools/minicpm_v47/draft_sweep_convert.sh 4.00 -- -cd mix.safetensors          # extra convert.py args after `--`
#   DSPARK_PREFIX=dspark-cal tools/minicpm_v47/draft_sweep_convert.sh 4.00 -- -cd target_taps.safetensors   # later arms
#
# Output: $DSPARK_OUT_ROOT/<prefix>-<bpw>bpw (default prefix `dspark`, bpw formatted %.2f). Existing packs are skipped unless
# FORCE=1. Environment (defaults in brackets):
#   DSPARK_SRC        bf16 source directory       [./dspark-hf]
#   DSPARK_OUT_ROOT   pack root                   [./draft-packs]
#   DSPARK_WORK_ROOT  scratch (qtensors, logs)    [./work/draft-quant]
#   DSPARK_PREFIX     pack name prefix            [dspark]
#   PYTHON            interpreter                 [python3]
#   GPU_LOCK          optional lock file          [unset: no lock]   (when set, conversion and verification run under
#                                                 a shared `flock -s -w LOCK_WAIT $GPU_LOCK`)
#   LOCK_WAIT         seconds to wait for it      [3600]
#   DEVICE            CUDA device index           [0]
#   VERIFY            1: run verify_draft_pack.py after each conversion (inventory, sha256 of verbatim tensors, codebook
#                     marker, per-matrix rel_l2, finite forward vs the bf16 draft), report -> <pack>/verify_report.json [1]
#   DSPARK_TARGET_DIR bf16 target HF dir, real embedding rows for the verify forward [unset: random embeddings]
#   BASE_REF          git ref the library patch hash is taken against [origin/gb10]
#
# Notes that matter:
#  - `-cb mul1` is required (every quantized linear carries a `.mul1` marker tensor that tells a loader which codebook
#    to use); it is passed explicitly although it is the default.
#  - no -hq / -hb / -vb: the draft has no LM head and no vision tower. `--out_scales` stays at the fork default (always).
#  - convert.py is run by absolute path and without `-u`; PYTHONUNBUFFERED=1 gives unbuffered output instead.
#  - uncalibrated: the architecture advertises caps["uncalibrated_quantize"], so no tokenizer and no calibration data are
#    needed (identity Hessian). The calibrated arm is a TODO of exllamav3/architecture/qwen3_dspark.py.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FORK="$(cd "$HERE/../.." && pwd)"
SRC="${DSPARK_SRC:-./dspark-hf}"
OUT_ROOT="${DSPARK_OUT_ROOT:-./draft-packs}"
WORK_ROOT="${DSPARK_WORK_ROOT:-./work/draft-quant}"
PREFIX="${DSPARK_PREFIX:-dspark}"
PYTHON="${PYTHON:-python3}"
LOCK="${GPU_LOCK:-}"
LOCK_WAIT="${LOCK_WAIT:-3600}"
DEVICE="${DEVICE:-0}"
VERIFY="${VERIFY:-1}"
TARGET_DIR="${DSPARK_TARGET_DIR:-}"
BASE_REF="${BASE_REF:-origin/gb10}"

# Run a command under the shared GPU lock when GPU_LOCK is set, plainly otherwise
run_locked() {
    if [ -n "$LOCK" ]; then flock -s -w "$LOCK_WAIT" "$LOCK" "$@"; else "$@"; fi
}
if [ -n "$LOCK" ]; then LOCK_DESC="flock -s -w $LOCK_WAIT $LOCK"; else LOCK_DESC="none"; fi

BPWS=()
EXTRA=()
while [ $# -gt 0 ]; do
    case "$1" in
        --) shift; EXTRA=("$@"); break ;;
        -h|--help) sed -n '2,/^set -euo pipefail$/{/^set -euo pipefail$/!p}' "${BASH_SOURCE[0]}"; exit 0 ;;
        *) BPWS+=("$1"); shift ;;
    esac
done
[ ${#BPWS[@]} -gt 0 ] || { echo "usage: $0 BPW [BPW ...] [-- extra convert.py args]" >&2; exit 2; }
[ -f "$SRC/config.json" ] || { echo "no config.json in $SRC" >&2; exit 2; }
[ -x "$PYTHON" ] || command -v "$PYTHON" >/dev/null 2>&1 || { echo "interpreter not found: $PYTHON" >&2; exit 2; }
# a built extension (exllamav3_ext*.so; a symlink or copy next to convert.py works) is required
ls "$FORK"/exllamav3_ext*.so >/dev/null 2>&1 || { echo "no exllamav3_ext*.so in $FORK" >&2; exit 2; }

mkdir -p "$OUT_ROOT" "$WORK_ROOT"

git_in_fork() { git -C "$FORK" "$@"; }
HEAD_COMMIT="$(git_in_fork rev-parse HEAD)"
BRANCH="$(git_in_fork rev-parse --abbrev-ref HEAD)"
BASE_COMMIT="$(git_in_fork merge-base HEAD "$BASE_REF" 2>/dev/null || echo "$HEAD_COMMIT")"
# the library patch: everything under exllamav3/ that differs from the base branch (the tools/ script is not part of it)
PATCH_SHA="$(git_in_fork diff "$BASE_COMMIT" HEAD -- exllamav3 | sha256sum | cut -d' ' -f1)"
SHIM_SHA="$(sha256sum "$FORK/exllamav3/architecture/qwen3_dspark.py" | cut -d' ' -f1)"
DIRTY="$(git_in_fork status --porcelain --untracked-files=no -- exllamav3 convert.py | wc -l)"

for bpw_in in "${BPWS[@]}"; do
    B="$(printf '%.2f' "$bpw_in")"
    NAME="${PREFIX}-${B}bpw"
    OUT="$OUT_ROOT/$NAME"
    WORK="$WORK_ROOT/$NAME"
    LOG="$WORK_ROOT/$NAME.convert.log"
    TIMEF="$WORK_ROOT/$NAME.time"
    if [ -f "$OUT/model.safetensors" ] && [ "${FORCE:-0}" != "1" ]; then
        echo "[$NAME] exists, skipping (FORCE=1 to redo)"
        continue
    fi
    rm -rf "$WORK" "$OUT" "$TIMEF"
    mkdir -p "$WORK"

    CMD=( "$PYTHON" "$FORK/convert.py" -i "$SRC" -o "$OUT" -w "$WORK" -b "$B" -cb mul1 -d 0 "${EXTRA[@]}" )
    printf -v CMD_STR '%q ' env CUDA_VISIBLE_DEVICES="$DEVICE" PYTHONUNBUFFERED=1 "${CMD[@]}"
    echo "[$NAME] $CMD_STR"

    T_LOCK0="$(date +%s.%N)"
    set +e
    run_locked bash -c '
        timef="$1"; shift
        s=$(date +%s.%N)
        /usr/bin/time -v -o "$timef.rss" env CUDA_VISIBLE_DEVICES='"$DEVICE"' PYTHONUNBUFFERED=1 "$@"
        rc=$?
        e=$(date +%s.%N)
        echo "$s $e $rc" > "$timef"
        exit $rc
    ' _ "$TIMEF" "${CMD[@]}" > "$LOG" 2>&1
    RC=$?
    set -e
    if [ $RC -ne 0 ] || [ ! -f "$OUT/model.safetensors" ]; then
        echo "[$NAME] FAILED rc=$RC, see $LOG" >&2
        tail -n 15 "$LOG" >&2 || true
        exit 1
    fi
    read -r T0 T1 _ < "$TIMEF"
    WALL="$(awk -v a="$T0" -v b="$T1" 'BEGIN{printf "%.1f", b-a}')"
    LOCKW="$(awk -v a="$T_LOCK0" -v b="$T0" 'BEGIN{printf "%.1f", b-a}')"
    PEAK_KB="$(awk -F': ' '/Maximum resident set size/{print $2}' "$TIMEF.rss" 2>/dev/null || echo "?")"
    cp "$LOG" "$OUT/convert.log"
    {
        echo "pack:            $NAME"
        echo "source:          $SRC"
        echo "source revision: openbmb/MiniCPM-V-4.7-35B-A3B-DSpark 5facb75c251ffd524524a544c1f5fd97ba0d6c5c (bf16, 389,223,169 params)"
        echo "source sha256:   $(sha256sum "$SRC/model.safetensors" | cut -d' ' -f1)  model.safetensors"
        echo "command:         $CMD_STR"
        echo "lock:            $LOCK_DESC"
        echo "target bpw:      $B (uncalibrated, identity Hessian; codebook mul1; out_scales default = always; no head, no vision)"
        echo "extra args:      ${EXTRA[*]:-(none)}"
        echo "fork:            $(git_in_fork remote get-url origin 2>/dev/null || echo local) branch $BRANCH"
        echo "fork commit:     $HEAD_COMMIT (merge-base with $BASE_REF: $BASE_COMMIT), library dirty files: $DIRTY"
        echo "patch sha256:    $PATCH_SHA  (git diff $BASE_COMMIT HEAD -- exllamav3/)"
        echo "shim sha256:     $SHIM_SHA  exllamav3/architecture/qwen3_dspark.py"
        echo "python:          $("$PYTHON" -c 'import sys,torch;print(sys.version.split()[0],"torch",torch.__version__)')"
        echo "system:          $(uname -sr)"
        echo "started:         $(date -u -d "@${T0%.*}" +%Y-%m-%dT%H:%M:%SZ)"
        echo "wall-clock:      ${WALL} s (conversion only; lock wait ${LOCKW} s)"
        echo "peak host RSS:   ${PEAK_KB} kB (time -v)"
        echo "pack bytes:      $(du -sb "$OUT" | cut -f1) total, $(stat -c %s "$OUT/model.safetensors") model.safetensors"
    } > "$OUT/CONVERT.txt"
    echo "[$NAME] done in ${WALL} s -> $OUT"

    if [ "$VERIFY" != "0" ]; then
        VARGS=( "$HERE/verify_draft_pack.py" --pack "$OUT" --source "$SRC" --json "$OUT/verify_report.json" --device cuda:0 )
        if [ -n "$TARGET_DIR" ] && [ -d "$TARGET_DIR" ]; then VARGS+=( --target-dir "$TARGET_DIR" ); fi
        set +e
        run_locked env CUDA_VISIBLE_DEVICES="$DEVICE" PYTHONUNBUFFERED=1 "$PYTHON" "${VARGS[@]}" \
            > "$OUT/verify.log" 2>&1
        VRC=$?
        set -e
        tail -n 1 "$OUT/verify.log"
        [ $VRC -eq 0 ] || { echo "[$NAME] VERIFY FAILED, see $OUT/verify.log" >&2; exit 1; }
    fi
done
