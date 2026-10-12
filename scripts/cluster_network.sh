#!/usr/bin/env bash
# Source once per node before torchrun, after setting MASTER_ADDR and PYTHON_BIN.
# Shared by every multi-node recipe; no fixed node/GPU count or training settings.
timejev_configure_cluster_network() {
  local network_dir network_python socket_ifname hca_index
  local -a hca_list=()
  network_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  network_python="${PYTHON_BIN:-${TIMEJEV_PYTHON:-python}}"
  if [[ "${TIMEJEV_ROUTE_SOCKET:-1}" == 1 ]]; then
    socket_ifname="$("$network_python" -B "$network_dir/cluster_socket_interface.py" \
      --master "${MASTER_ADDR:?Set MASTER_ADDR before cluster_network.sh}" \
      --preferred "${NCCL_SOCKET_IFNAME:-}")" || return
    export NCCL_SOCKET_IFNAME="=$socket_ifname"
    export GLOO_SOCKET_IFNAME="$socket_ifname"
  else
    if [[ -z "${NCCL_SOCKET_IFNAME:-}" && -d /sys/class/net/bond0 ]]; then
      export NCCL_SOCKET_IFNAME=bond0
    fi
    if [[ -z "${GLOO_SOCKET_IFNAME:-}" && -n "${NCCL_SOCKET_IFNAME:-}" ]]; then
      export GLOO_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME#=}"
    fi
  fi
  if [[ -z "${NCCL_IB_HCA:-}" ]]; then
    for hca_index in {2..9}; do
      [[ ! -d "/sys/class/infiniband/mlx5_bond_${hca_index}" ]] || hca_list+=("mlx5_bond_${hca_index}")
    done
    if ((${#hca_list[@]})); then export NCCL_IB_HCA="$(IFS=,; echo "${hca_list[*]}")"; fi
  fi
}
timejev_configure_cluster_network
