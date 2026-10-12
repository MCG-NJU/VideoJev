"""Select an addressed socket interface without opening a distributed test job.

Keep a usable platform preference; otherwise use the kernel route to MASTER_ADDR.
UDP connect only queries local routing and does not send network packets.
This affects NCCL bootstrap/Gloo sockets, not NCCL's RDMA transport or HCA list.
"""
import argparse
import fcntl
import socket
import struct
import time


def ipv4_interfaces():
    interfaces = {}
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        for _, name in socket.if_nameindex():
            request = struct.pack('256s', name.encode()[:15])
            try:
                flags = struct.unpack_from('H', fcntl.ioctl(sock, 0x8913, request), 16)[0]
                address = socket.inet_ntoa(fcntl.ioctl(sock, 0x8915, request)[20:24])
            except OSError:
                continue
            if flags & 1 and not flags & 8 and address != '0.0.0.0':
                interfaces[name] = address
    return interfaces


def preferred_interfaces(interfaces, preference):
    exclude = preference.startswith('^')
    pattern = preference[1:] if exclude else preference
    exact = pattern.startswith('=')
    patterns = (pattern[1:] if exact else pattern).split(',')
    patterns = [p for p in patterns if p]
    matches = [name for name in interfaces if any(
        name == p if exact else name.startswith(p) for p in patterns)]
    return [name for name in interfaces if name not in matches] if exclude else matches


def route_source(master):
    address = socket.getaddrinfo(master, None, socket.AF_INET, socket.SOCK_DGRAM)[0][4][0]
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        sock.connect((address, 9))
        return sock.getsockname()[0]


def select_interface(interfaces, preference, source):
    candidates = preferred_interfaces(interfaces, preference) if preference else []
    routed = [name for name, address in interfaces.items() if address == source]
    if candidates:
        return next((name for name in routed if name in candidates), candidates[0])
    if routed:
        return routed[0]
    raise RuntimeError(f'No addressed interface matches the route source {source}; '
                       f'available interfaces: {interfaces}')


def choose_interface(master, preference, dns_timeout=30):
    interfaces = ipv4_interfaces()
    candidates = preferred_interfaces(interfaces, preference) if preference else []
    if len(candidates) == 1:
        # The usual platform bond0 needs no DNS lookup or extra startup wait.
        return candidates[0]
    deadline = time.monotonic() + dns_timeout
    while True:
        try:
            source = route_source(master)
            break
        except socket.gaierror:
            # A newly created master DNS record can appear a few seconds after
            # its workers. Match torchrun's tolerance of this transient state.
            if time.monotonic() >= deadline:
                raise
            time.sleep(min(1, max(0, deadline - time.monotonic())))
    return select_interface(interfaces, preference, source)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--master', required=True)
    parser.add_argument('--preferred', default='')
    args = parser.parse_args()
    print(choose_interface(args.master, args.preferred))


if __name__ == '__main__':
    main()
