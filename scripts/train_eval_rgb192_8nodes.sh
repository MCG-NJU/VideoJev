#!/usr/bin/env bash
# Run once per allocated node; parse the whole workflow before long-running work.
video_jev_main() {
set -euo pipefail
local SCRIPT_DIR PROJECT MODE NODE_INDEX PYTHON_BIN SOURCE STAGE ACTION
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT="$(cd "$SCRIPT_DIR/.." && pwd)"
MODE="${1:-train}"
[[ $# -le 1 && ( "$MODE" == train || "$MODE" == --dry-run ) ]] || { echo "Usage: $0 [--dry-run]" >&2; exit 2; }
fail() { echo "ERROR: $*" >&2; exit 2; }
[[ "${NNODES:-${NODE_COUNT:-${SLURM_JOB_NUM_NODES:-8}}}" == 8 ]] || fail 'Requires 8 nodes.'
[[ "${NPROC_PER_NODE:-${GPUS_PER_NODE:-8}}" == 8 ]] || fail 'Requires 8 GPUs per node.'
NODE_INDEX="${NODE_RANK:-${SLURM_NODEID:-${RANK:-}}}"
[[ "$NODE_INDEX" =~ ^[0-7]$ ]] || fail 'Set NODE_RANK=0..7 (or platform RANK/SLURM_NODEID).'
[[ -z "${LOCAL_RANK:-}" ]] || fail 'Run this command once per node, outside torchrun workers.'
[[ -z "${SLURM_NTASKS_PER_NODE:-}" || "$SLURM_NTASKS_PER_NODE" == 1 ]] || fail 'Use one Slurm task per node.'
if [[ -z "${MASTER_ADDR:-}" && -n "${SLURM_JOB_NODELIST:-}" ]]; then
  local MASTER_NODES
  MASTER_NODES="$(scontrol show hostnames "$SLURM_JOB_NODELIST")"
  export MASTER_ADDR="${MASTER_NODES%%$'\n'*}"
fi
[[ -n "${MASTER_ADDR:-}" ]] || fail 'Set MASTER_ADDR to the reachable address of node 0.'
case "$MASTER_ADDR" in localhost|localhost.*|127.*|::1|0.0.0.0) fail 'MASTER_ADDR must be reachable across all 8 nodes.';; esac
export MASTER_PORT="${MASTER_PORT:-29690}"
[[ "$MASTER_PORT" =~ ^[1-9][0-9]{0,4}$ ]] && ((MASTER_PORT<=65535)) || fail 'Invalid MASTER_PORT.'
export WORK_DIR="${WORK_DIR:-$PROJECT/runs/vtg_qa_rgb192_zero3_gbs64_8nodes_v1}"
[[ "$WORK_DIR" == /* && "$WORK_DIR" != "$PROJECT" ]] || fail 'WORK_DIR must be an absolute shared run directory.'
PYTHON_BIN="${TIMEJEV_PYTHON:-python}"
local -a TORCHRUN=("$PYTHON_BIN" -B -m torch.distributed.run --nnodes=8 --nproc-per-node=8
  --node-rank="$NODE_INDEX" --master-addr="$MASTER_ADDR" --master-port="$MASTER_PORT" --max-restarts=0)
echo "8 nodes x 8 GPUs; GBS64 packed sequences; 196608 tokens/pack; online RGB; ZeRO-3; accumulation=1; node=$NODE_INDEX"
echo 'head: VTG BCE / QA CE; joint: VTG BCE+2*SoftIoU / QA CE'
if [[ "$MODE" == --dry-run ]]; then
  printf '%q ' "$PYTHON_BIN" scripts/prepare_rgb192.py --project "$PROJECT" --out "$WORK_DIR/data"
  printf '\n'
  for STAGE in head_only joint; do
    printf '%q ' "${TORCHRUN[@]}" scripts/train.py --config "$WORK_DIR/source/configs/$STAGE.json" --out "$WORK_DIR/$STAGE"
    if [[ "$STAGE" == joint ]]; then printf '%q ' --init-head "$WORK_DIR/head_only/head.pt"; fi
    printf '\n'
  done
  printf '%q ' "${TORCHRUN[@]}" scripts/evaluate_rgb192.py --checkpoint "$WORK_DIR/joint/hf_checkpoint" --out "$WORK_DIR/evaluation"
  printf '\nauto resume; skip completed stages; verify all seven benchmarks before cleanup\n'
  return
fi
mkdir -p "$WORK_DIR/logs"
exec 9>"$WORK_DIR/.node${NODE_INDEX}.lock"
flock -n 9 || fail "Node rank $NODE_INDEX already runs this pipeline."
exec > >(tee -a "$WORK_DIR/logs/node${NODE_INDEX}-$(date -u +%Y%m%dT%H%M%S)-$$.log" 9>&-) 2>&1
trap 'video_jev_exit=$?; echo "pipeline_exit=$video_jev_exit"' EXIT
SOURCE="$("$PYTHON_BIN" -B "$SCRIPT_DIR/pipeline.py" snapshot --profile rgb192 --project "$PROJECT" --out "$WORK_DIR")"
cd "$SOURCE"
export TIMEJEV_USE_NODE_LOCAL_RUNTIME=0
source "$SOURCE/scripts/env.sh"
export PYTHONDONTWRITEBYTECODE=1
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_TRACE_BUFFER_SIZE=2000
export TORCH_NCCL_DUMP_ON_TIMEOUT=1
source "$SOURCE/scripts/cluster_cuda128.sh"
source "$SOURCE/scripts/cluster_network.sh"
"$PYTHON_BIN" -B scripts/prepare_rgb192.py --project "$PROJECT" --out "$WORK_DIR/data" --workers "${RGB_METADATA_WORKERS:-12}"
for STAGE in head_only joint; do
  ACTION="$("$PYTHON_BIN" -B scripts/pipeline.py action --out "$WORK_DIR" --stage "$STAGE")"
  if [[ "$ACTION" == skip ]]; then echo "$STAGE already complete"; continue; fi
  # Metadata-only disk guard; no duplicate model/optimizer diagnostic at launch.
  "$PYTHON_BIN" -B - "$WORK_DIR" <<'PY'
import shutil,sys
free=shutil.disk_usage(sys.argv[1]).free
if free < 700*10**9: raise SystemExit(f'At least 700 GB free required; available {free/1e9:.1f} GB')
PY
  local -a EXTRA=()
  if [[ "$ACTION" == resume ]]; then EXTRA+=(--resume "$WORK_DIR/$STAGE/safety_deepspeed"); fi
  if [[ "$STAGE" == joint ]]; then EXTRA+=(--init-head "$WORK_DIR/head_only/head.pt"); fi
  "${TORCHRUN[@]}" scripts/train.py --config "$SOURCE/configs/$STAGE.json" --out "$WORK_DIR/$STAGE" "${EXTRA[@]}"
done
"${TORCHRUN[@]}" scripts/evaluate_rgb192.py --checkpoint "$WORK_DIR/joint/hf_checkpoint" --out "$WORK_DIR/evaluation"
if [[ "$NODE_INDEX" == 0 ]]; then
  "$PYTHON_BIN" -B scripts/pipeline.py cleanup --out "$WORK_DIR"
fi
}
video_jev_main "$@"
