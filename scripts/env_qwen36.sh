#!/usr/bin/env bash
# Qwen3.6 entrypoint only. Do not alter the pinned 4B environment or its imports.
QWEN36_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
QWEN36_PROJECT="$(cd "$QWEN36_SCRIPT_DIR/.." && pwd)"
# Reuse env.sh's existing identical node-local runtime when available.
# The independent extension overlay below retains highest import priority.
source "$QWEN36_SCRIPT_DIR/env.sh"
export PYTHONPATH="$QWEN36_PROJECT/.deps-qwen36:$PYTHONPATH"
if [[ -n "${TIMEJEV_QWEN36_ENV:-}" ]]; then
  export PATH="$TIMEJEV_QWEN36_ENV/bin:$PATH"
  export TIMEJEV_QWEN36_PYTHON="${TIMEJEV_QWEN36_PYTHON:-$TIMEJEV_QWEN36_ENV/bin/python}"
fi
export TIMEJEV_QWEN36_PYTHON="${TIMEJEV_QWEN36_PYTHON:-$TIMEJEV_PYTHON}"
export PYTHONDONTWRITEBYTECODE=1
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export TORCH_NCCL_TRACE_BUFFER_SIZE="${TORCH_NCCL_TRACE_BUFFER_SIZE:-2000}"
export TORCH_NCCL_DUMP_ON_TIMEOUT="${TORCH_NCCL_DUMP_ON_TIMEOUT:-1}"
# All expert kernels are XTuner grouped kernels; absence is a hard error in the
# Qwen3.6 adapter, never a Python-loop fallback. Keep FP8 and MTP disabled.
export TIMEJEV_QWEN36_EXPERT_BACKEND=xtuner_grouped_gemm
export FLA_INTRACARD_CP=0
