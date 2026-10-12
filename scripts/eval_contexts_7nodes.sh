#!/usr/bin/env bash
# Invoke once per node. Parse the entire workflow before long-running commands.
video_jev_eval_main() {
set -euo pipefail
local SCRIPT_DIR PROJECT MODE NODE_INDEX PYTHON_BIN SOURCE
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT="$(cd "$SCRIPT_DIR/.." && pwd)"
MODE="${1:-evaluate}"
[[ $# -le 1 && ( "$MODE" == evaluate || "$MODE" == --dry-run ) ]] || { echo "Usage: $0 [--dry-run]" >&2; exit 2; }
fail() { echo "ERROR: $*" >&2; exit 2; }
[[ "${NNODES:-${NODE_COUNT:-${SLURM_JOB_NUM_NODES:-7}}}" == 7 ]] || fail 'Requires 7 nodes.'
[[ "${NPROC_PER_NODE:-${GPUS_PER_NODE:-8}}" == 8 ]] || fail 'Requires 8 GPUs per node.'
NODE_INDEX="${NODE_RANK:-${SLURM_NODEID:-${RANK:-}}}"
[[ "$NODE_INDEX" =~ ^[0-6]$ ]] || fail 'Set NODE_RANK=0..6 (or platform RANK/SLURM_NODEID).'
[[ -z "${LOCAL_RANK:-}" ]] || fail 'Run once per node, outside torchrun workers.'
[[ -z "${SLURM_NTASKS_PER_NODE:-}" || "$SLURM_NTASKS_PER_NODE" == 1 ]] || fail 'Use one Slurm task per node.'
if [[ -z "${MASTER_ADDR:-}" && -n "${SLURM_JOB_NODELIST:-}" ]]; then
  local MASTER_NODES
  MASTER_NODES="$(scontrol show hostnames "$SLURM_JOB_NODELIST")"
  export MASTER_ADDR="${MASTER_NODES%%$'\n'*}"
fi
[[ -n "${MASTER_ADDR:-}" ]] || fail 'Set MASTER_ADDR to the reachable address of node 0.'
case "$MASTER_ADDR" in localhost|localhost.*|127.*|::1|0.0.0.0) fail 'MASTER_ADDR must be reachable across all 7 nodes.';; esac
export MASTER_PORT="${MASTER_PORT:-29710}"
[[ "$MASTER_PORT" =~ ^[1-9][0-9]{0,4}$ ]] && ((MASTER_PORT<=65535)) || fail 'Invalid MASTER_PORT.'
export CHECKPOINT="${CHECKPOINT:-$PROJECT/runs/videojev4b/joint/hf_checkpoint}"
export WORK_DIR="${WORK_DIR:-$PROJECT/runs/qa_rgb_context80_196608_262144_7nodes_v1}"
export CONTEXT_PLANS="${CONTEXT_PLANS:-$PROJECT/data/context_eval_v1}"
[[ "$WORK_DIR" == /* && "$WORK_DIR" != "$PROJECT" ]] || fail 'WORK_DIR must be an absolute shared output directory.'
[[ -f "$CHECKPOINT" || -f "$CHECKPOINT/export_complete.json" ]] || fail "Checkpoint not found: $CHECKPOINT"
PYTHON_BIN="${TIMEJEV_PYTHON:-python}"
local -a TORCHRUN=("$PYTHON_BIN" -B -m torch.distributed.run --nnodes=7 --nproc-per-node=8
  --node-rank="$NODE_INDEX" --master-addr="$MASTER_ADDR" --master-port="$MASTER_PORT" --max-restarts=0)
echo "7 nodes x 8 GPUs; node=$NODE_INDEX; contexts=196608,262144; visual budgets=157286,209715"
echo 'LVBench / Video-MME / LongVideoBench validation / MLVU Dev MC; 2fps; 2048 frames; max pixels=640*640'
echo "checkpoint=$CHECKPOINT; output=$WORK_DIR/evaluation"
if [[ "$MODE" == --dry-run ]]; then
  printf '%q ' "${TORCHRUN[@]}" -m scripts.evaluate_contexts --checkpoint "$CHECKPOINT" \
    --plans "$WORK_DIR/source/plans" --out "$WORK_DIR/evaluation" --workers "${RGB_WORKERS:-1}"
  printf '\nOne model load per GPU; sequential contexts; verified per-video/benchmark resume.\n'
  return
fi
mkdir -p "$WORK_DIR/logs"
exec 9>"$WORK_DIR/.node${NODE_INDEX}.lock"
flock -n 9 || fail "Node rank $NODE_INDEX already runs this evaluation."
exec > >(tee -a "$WORK_DIR/logs/node${NODE_INDEX}-$(date -u +%Y%m%dT%H%M%S)-$$.log" 9>&-) 2>&1
trap 'video_jev_exit=$?; echo "evaluation_exit=$video_jev_exit"' EXIT
cd "$PROJECT"
export TIMEJEV_USE_NODE_LOCAL_RUNTIME=0
source "$PROJECT/scripts/env.sh"
SOURCE="$("$PYTHON_BIN" -B -m scripts.context_eval_pipeline --project "$PROJECT" --out "$WORK_DIR" \
  --plans "$CONTEXT_PLANS" --checkpoint "$CHECKPOINT")"
cd "$SOURCE"
source "$SOURCE/scripts/env.sh"
export PYTHONDONTWRITEBYTECODE=1
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_TRACE_BUFFER_SIZE=2000
export TORCH_NCCL_DUMP_ON_TIMEOUT=1
source "$SOURCE/scripts/cluster_cuda128.sh"
source "$SOURCE/scripts/cluster_network.sh"
"${TORCHRUN[@]}" -m scripts.evaluate_contexts --checkpoint "$CHECKPOINT" --plans "$SOURCE/plans" \
  --out "$WORK_DIR/evaluation" --workers "${RGB_WORKERS:-1}"
}
video_jev_eval_main "$@"
