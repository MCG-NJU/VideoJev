import pytest

from scripts.cluster_socket_interface import select_interface


def test_missing_or_unaddressed_bond_uses_routed_interface():
    assert select_interface({'eth0': '172.20.99.254'}, 'bond0', '172.20.99.254') == 'eth0'


def test_valid_bond_preference_preserved():
    assert select_interface({'eth0': '172.20.99.254', 'bond0': '10.0.0.2'},
                            'bond0', '172.20.99.254') == 'bond0'


@pytest.mark.parametrize('preference', ['', '=eth0', 'eth', '^docker,lo'])
def test_master_self_route_uses_addressed_interface_not_loopback(preference):
    assert select_interface({'eth0': '172.20.99.254'}, preference, '172.20.99.254') == 'eth0'


def test_unknown_route_fails_without_guessing():
    with pytest.raises(RuntimeError, match='No addressed interface'):
        select_interface({'eth0': '172.20.99.254'}, 'bond0', '10.0.0.3')


def test_exact_name_does_not_select_prefix_collision():
    interfaces = {'eth01': '10.0.0.1', 'eth0': '10.0.0.2'}
    assert select_interface(interfaces, '=eth0', '10.0.0.1') == 'eth0'


def test_valid_platform_interface_does_not_wait_for_dns(monkeypatch):
    from scripts import cluster_socket_interface as module
    monkeypatch.setattr(module, 'ipv4_interfaces', lambda: {'bond0': '10.0.0.2', 'eth0': '172.20.1.2'})
    def unexpected_dns(master):
        raise AssertionError('An unambiguous usable preference does not need DNS')
    monkeypatch.setattr(module, 'route_source', unexpected_dns)
    assert module.choose_interface('not-yet-published-master', 'bond0') == 'bond0'


def test_missing_platform_interface_retries_transient_master_dns(monkeypatch):
    from scripts import cluster_socket_interface as module
    import socket
    monkeypatch.setattr(module, 'ipv4_interfaces', lambda: {'eth0': '172.20.1.2'})
    attempts = []
    def route(master):
        attempts.append(master)
        if len(attempts) == 1:
            raise socket.gaierror(-2, 'Master record not yet published')
        return '172.20.1.2'
    monkeypatch.setattr(module, 'route_source', route)
    monkeypatch.setattr(module.time, 'sleep', lambda seconds: None)
    assert module.choose_interface('new-master', 'bond0') == 'eth0'
    assert len(attempts) == 2


def test_persistent_dns_error_is_bounded(monkeypatch):
    from scripts import cluster_socket_interface as module
    import socket
    monkeypatch.setattr(module, 'ipv4_interfaces', lambda: {'eth0': '172.20.1.2'})
    def route(master):
        raise socket.gaierror(-2, 'Unknown master')
    monkeypatch.setattr(module, 'route_source', route)
    with pytest.raises(socket.gaierror):
        module.choose_interface('unknown-master', 'bond0', dns_timeout=0)


@pytest.mark.parametrize('nodes', [2, 4, 8, 16])
def test_shared_network_configuration_is_independent_of_node_count(tmp_path, monkeypatch, nodes):
    import os
    import subprocess
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    fake = tmp_path/'python'
    fake.write_text('#!/bin/sh\necho eth0\n')
    fake.chmod(0o755)
    env = dict(os.environ, PYTHON_BIN=str(fake), MASTER_ADDR='allocated-master', NNODES=str(nodes),
               TIMEJEV_ROUTE_SOCKET='1', NCCL_SOCKET_IFNAME='bond0', GLOO_SOCKET_IFNAME='bond0',
               NCCL_IB_HCA='mlx5_bond_2,mlx5_bond_3', NCCL_IB_DISABLE='0', NCCL_NET='IB')
    result = subprocess.run(['bash', '-c',
        'set -euo pipefail; source "$1"; printf "%s\\n" "$NNODES" "$NCCL_SOCKET_IFNAME" '
        '"$GLOO_SOCKET_IFNAME" "$NCCL_IB_HCA" "$NCCL_IB_DISABLE" "$NCCL_NET"',
        'test', str(root/'scripts/cluster_network.sh')], env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == [str(nodes), '=eth0', 'eth0', 'mlx5_bond_2,mlx5_bond_3', '0', 'IB']
