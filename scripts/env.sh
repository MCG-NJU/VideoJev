#!/usr/bin/env bash
# Activate your Python environment before sourcing this file.
VIDEO_JEV_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export VIDEO_JEV_ROOT
export PYTHONPATH="$VIDEO_JEV_ROOT${PYTHONPATH:+:$PYTHONPATH}"
if [[ -n "${XTUNER_ROOT:-}" ]]; then
  export PYTHONPATH="$XTUNER_ROOT:$PYTHONPATH"
fi
export TIMEJEV_PYTHON="${TIMEJEV_PYTHON:-$(command -v python)}"
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-2}"
export TOKENIZERS_PARALLELISM=false
# Preserve native DeltaNet recurrence at long sequence lengths.
export FLA_INTRACARD_CP=0
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
