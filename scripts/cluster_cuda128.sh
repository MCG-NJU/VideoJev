#!/usr/bin/env bash
# Use the assembler shipped with the active Triton installation.
if [[ -z "${TRITON_PTXAS_PATH:-}" ]]; then
  TRITON_PTXAS_PATH="$("${TIMEJEV_PYTHON:-python}" -c 'import pathlib, triton; print(pathlib.Path(triton.__file__).parent / "backends/nvidia/bin/ptxas")')" || return 2
  export TRITON_PTXAS_PATH
fi
[[ -x "$TRITON_PTXAS_PATH" ]] || { echo "ERROR: Missing ptxas: $TRITON_PTXAS_PATH" >&2; return 2; }
VIDEOJEV_PTXAS_VERSION="$("$TRITON_PTXAS_PATH" --version)" || return 2
[[ "$VIDEOJEV_PTXAS_VERSION" == *'release 12.8,'* ]] || {
  echo "ERROR: This recipe requires CUDA 12.8 ptxas: $VIDEOJEV_PTXAS_VERSION" >&2
  return 2
}
