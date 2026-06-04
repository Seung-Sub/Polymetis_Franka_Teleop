#!/usr/bin/env bash
# run_eval_dit.sh -- deploy a DiT DP on the real Franka in one of 3 precisions.
# Usage:   bash bin/run_eval_dit.sh <task> <type> [extra eval args...]
#   <task> = peg | pap          <type> = fp32 | ptq | agiledp
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TASK="${1:?task: peg|pap}"; TYPE="${2:?type: fp32|ptq|agiledp}"; shift 2 || true

case "$TASK" in
  peg) FP="$ROOT/checkpoints/dp/dit_franka_peg_v2.1_epoch225.ckpt" ;;
  pap) FP="$ROOT/checkpoints/dp/dit_franka_pap_v2.1_epoch300.ckpt" ;;
  *) echo "task must be peg|pap"; exit 1 ;;
esac
QDIR="$ROOT/checkpoints/dp_quant/$TASK"
CALIB="$ROOT/checkpoints/dp_quant/DiffusionInput_calib_franka.pth"

# DiT output is intrinsically noisier per-step than UNet, so relax the
# degenerate-chunk guard (the per-step MAX_POS_STEP + workspace check stay).
export DEGENERATE_INTRA_STEP_M="${DEGENERATE_INTRA_STEP_M:-0.30}"

case "$TYPE" in
  fp32)    unset QUANT_CKPT QTYPE N_BIT CALIB_DATA ;;
  ptq)     export QUANT_CKPT="$QDIR/naiveq_w4a4.pth"      QTYPE=naiveq N_BIT=4 CALIB_DATA="$CALIB" ;;
  agiledp) export QUANT_CKPT="$QDIR/qalora_w4a4_150ep.pth" QTYPE=qalora N_BIT=4 CALIB_DATA="$CALIB" ;;
  *) echo "type must be fp32|ptq|agiledp"; exit 1 ;;
esac
echo "[run_eval_dit] task=$TASK type=$TYPE degen_thr=$DEGENERATE_INTRA_STEP_M FP=$FP ${QUANT_CKPT:+quant=$QUANT_CKPT qtype=$QTYPE}"
exec bash "$ROOT/bin/run_eval_dp.sh" "$FP" "$@"
