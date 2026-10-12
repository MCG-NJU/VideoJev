#!/usr/bin/env bash
# One invocation per node. Parse the whole workflow before long-running training.
video_jev_qwen36_main() {
set -euo pipefail
local SCRIPT_DIR PROJECT NODE_INDEX PYTHON_BIN SOURCE STAGE MODE
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT="$(cd "$SCRIPT_DIR/.." && pwd)"
MODE="${1:-train}"
fail() { echo "ERROR: $*" >&2; exit 2; }
[[ $# -le 1 && ( "$MODE" == train || "$MODE" == --dry-run ) ]] || fail "Usage: $0 [--dry-run]"
[[ "${NNODES:-${NODE_COUNT:-${SLURM_JOB_NUM_NODES:-8}}}" == 8 ]] || fail 'Requires 8 nodes.'
[[ "${NPROC_PER_NODE:-${GPUS_PER_NODE:-8}}" == 8 ]] || fail 'Requires 8 GPUs per node.'
NODE_INDEX="${NODE_RANK:-${SLURM_NODEID:-${RANK:-}}}"
[[ "$NODE_INDEX" =~ ^[0-7]$ ]] || fail 'Set NODE_RANK=0..7 (or platform RANK/SLURM_NODEID).'
[[ -z "${LOCAL_RANK:-}" ]] || fail 'Invoke once per node, outside torchrun workers.'
[[ -z "${SLURM_NTASKS_PER_NODE:-}" || "$SLURM_NTASKS_PER_NODE" == 1 ]] || fail 'Use one Slurm task per node.'
if [[ -z "${MASTER_ADDR:-}" && -n "${SLURM_JOB_NODELIST:-}" ]]; then
  local MASTER_NODES
  MASTER_NODES="$(scontrol show hostnames "$SLURM_JOB_NODELIST")"
  export MASTER_ADDR="${MASTER_NODES%%$'\n'*}"
fi
[[ -n "${MASTER_ADDR:-}" ]] || fail 'Set MASTER_ADDR to the reachable address of node 0.'
case "$MASTER_ADDR" in localhost|localhost.*|127.*|::1|0.0.0.0) fail 'MASTER_ADDR must be reachable across all nodes.';; esac
export MASTER_PORT="${MASTER_PORT:-29760}"
[[ "$MASTER_PORT" =~ ^[1-9][0-9]{0,4}$ ]] && ((MASTER_PORT<=65535)) || fail 'Invalid MASTER_PORT.'
export WORK_DIR="${WORK_DIR:-$PROJECT/runs/qwen36_vtg_qa_llm_lr2e6_gbs64_8nodes_v1}"
[[ "$WORK_DIR" == /* && "$WORK_DIR" != "$PROJECT" ]] || fail 'WORK_DIR must be an absolute shared run directory.'
PYTHON_BIN="${TIMEJEV_QWEN36_PYTHON:-${TIMEJEV_PYTHON:-python}}"
local -a TORCHRUN=("$PYTHON_BIN" -B -m torch.distributed.run --nnodes=8 --nproc-per-node=8
  --node-rank="$NODE_INDEX" --master-addr="$MASTER_ADDR" --master-port="$MASTER_PORT" --max-restarts=0)
echo "Qwen3.6-35B-A3B; 8x8 GPUs; GBS64 packs; train 147456; node=$NODE_INDEX"
echo 'Fresh base -> 30% head (VTG BCE / QA CE) -> 70% joint (LLM 2e-6, head 1e-4)'
echo 'Then grounding 147456/4fps; four QA benchmarks at 262144/2fps; all visual budgets 80%.'
if [[ "$MODE" == --dry-run ]]; then
  for STAGE in head_only joint; do
    printf '%q ' "${TORCHRUN[@]}" "$WORK_DIR/source/scripts/train_qwen36.py" \
      --config "$WORK_DIR/source/configs/qwen36_$STAGE.json" --out "$WORK_DIR/$STAGE" --resume auto
    if [[ "$STAGE" == joint ]]; then printf '%q ' --init-head "$WORK_DIR/head_only/head.pt"; fi
    printf '\n'
  done
  printf '%q ' "${TORCHRUN[@]}" "$WORK_DIR/source/scripts/evaluate_qwen36.py" \
    --checkpoint "$WORK_DIR/joint/hf_checkpoint" --plans "$WORK_DIR/data" --out "$WORK_DIR/evaluation"
  printf '\n'
  return
fi
mkdir -p "$WORK_DIR/logs"
exec 9>"$WORK_DIR/.node${NODE_INDEX}.lock"
flock -n 9 || fail "Node rank $NODE_INDEX already runs this pipeline."
exec > >(tee -a "$WORK_DIR/logs/node${NODE_INDEX}-$(date -u +%Y%m%dT%H%M%S)-$$.log" 9>&-) 2>&1
trap 'video_jev_exit=$?; echo "pipeline_exit=$video_jev_exit"' EXIT
export TIMEJEV_USE_NODE_LOCAL_RUNTIME=0
source "$SCRIPT_DIR/env_qwen36.sh"
# Preparation is metadata-only and lock-protected; the normal delivered launch reuses it.
if [[ ! -f "$WORK_DIR/source/snapshot.json" && ! -f "$PROJECT/data/qwen36_mixed_v1/ready.json" ]]; then
  "$PYTHON_BIN" -B "$SCRIPT_DIR/prepare_qwen36_mixed.py" --project "$PROJECT" \
    --out "$PROJECT/data/qwen36_mixed_v1" --workers "${RGB_METADATA_WORKERS:-8}"
fi
SOURCE="$("$PYTHON_BIN" -B "$SCRIPT_DIR/qwen36_pipeline.py" snapshot --project "$PROJECT" --out "$WORK_DIR")"
cd "$SOURCE"
source "$SOURCE/scripts/env_qwen36.sh"
source "$SOURCE/scripts/cluster_cuda128.sh"
source "$SOURCE/scripts/cluster_network.sh"
for STAGE in head_only joint; do
  if [[ "$("$PYTHON_BIN" -B scripts/qwen36_pipeline.py action --out "$WORK_DIR" --stage "$STAGE")" == skip ]]; then
    echo "$STAGE already complete"
    continue
  fi
  "$PYTHON_BIN" -B - "$WORK_DIR" <<'PY'
import shutil,sys
free=shutil.disk_usage(sys.argv[1]).free
if free < 1500*10**9: raise SystemExit(f'Need 1.5 TB free for sharded optimizer saves/export; available {free/1e9:.1f} GB')
PY
  local -a EXTRA=()
  if [[ "$STAGE" == joint ]]; then EXTRA+=(--init-head "$WORK_DIR/head_only/head.pt"); fi
  "${TORCHRUN[@]}" scripts/train_qwen36.py --config "$SOURCE/configs/qwen36_$STAGE.json" \
    --out "$WORK_DIR/$STAGE" --resume auto "${EXTRA[@]}"
done
"${TORCHRUN[@]}" scripts/evaluate_qwen36.py --checkpoint "$WORK_DIR/joint/hf_checkpoint" \
  --plans "$WORK_DIR/data" --out "$WORK_DIR/evaluation"
if [[ "$NODE_INDEX" == 0 ]]; then
  "$PYTHON_BIN" -B scripts/qwen36_pipeline.py finalize --out "$WORK_DIR"
  echo "All seven results: $WORK_DIR/RESULTS.md"
fi
}
video_jev_qwen36_main "$@"
