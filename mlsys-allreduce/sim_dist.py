import os
import sys
import time
import socket
import struct
import torch
import numpy as np

# Global state for current rank
_rank = 0
_world_size = 1
_listen_sock = None
_send_socks = {}  # dst -> socket
_recv_socks = {}  # src -> socket
_base_port = 16000
_sim_latency_s = float(os.environ.get("SIM_LATENCY_S", 0.0))
_sim_bw_mbps = float(os.environ.get("SIM_BW_MBPS", 0.0))  # 0.0 means unthrottled

def init_process_group(backend=None, init_method=None, rank=0, world_size=1, base_port=16000):
    global _rank, _world_size, _listen_sock, _send_socks, _recv_socks, _base_port
    _rank = rank
    _world_size = world_size
    _base_port = base_port
    
    # Extract port from init_method if provided (e.g., tcp://127.0.0.1:6585)
    if init_method and "tcp://" in init_method:
        try:
            parts = init_method.split(":")
            _base_port = int(parts[-1])
        except Exception:
            pass

    # Rank port = _base_port + rank
    my_port = _base_port + _rank
    _listen_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    _listen_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    _listen_sock.bind(("127.0.0.1", my_port))
    _listen_sock.listen(_world_size)

    # Establish full-mesh peer-to-peer TCP connections
    # Phase 1: Connect to ranks with id < rank (or all other ranks)
    # To avoid race conditions, all ranks wait a fraction of a second, then connect
    # Each incoming connection sends a 4-byte header identifying who connected: src_rank
    
    import threading
    accepted_threads = []
    
    def accept_connections():
        needed = _world_size - 1
        received = 0
        while received < needed:
            conn, _ = _listen_sock.accept()
            # Read 4-byte header indicating connecting rank
            header = conn.recv(4)
            while len(header) < 4:
                header += conn.recv(4 - len(header))
            src_r = struct.unpack("!I", header)[0]
            _recv_socks[src_r] = conn
            received += 1

    t = threading.Thread(target=accept_connections, daemon=True)
    t.start()

    time.sleep(0.05 * _world_size)  # Small startup grace period

    # Connect outward to all other ranks
    for dst in range(_world_size):
        if dst == _rank:
            continue
        dst_port = _base_port + dst
        connected = False
        start_wait = time.time()
        while not connected:
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.connect(("127.0.0.1", dst_port))
                # Send my rank in header
                s.sendall(struct.pack("!I", _rank))
                _send_socks[dst] = s
                connected = True
            except ConnectionRefusedError:
                time.sleep(0.05)
                if time.time() - start_wait > 30.0:
                    raise TimeoutError(f"Rank {_rank} timed out connecting to rank {dst} on port {dst_port}")

    t.join(timeout=30.0)
    if len(_recv_socks) != _world_size - 1:
        raise RuntimeError(f"Rank {_rank} failed to accept connections from all peers")

    barrier()

def get_rank():
    return _rank

def get_world_size():
    return _world_size

def send(tensor, dst):
    if dst == _rank:
        return
    if dst not in _send_socks:
        raise RuntimeError(f"Rank {_rank} has no connection to dst={dst}")
    
    sock = _send_socks[dst]
    # Ensure contiguous numpy view
    arr = tensor.detach().cpu().numpy()
    if not arr.flags["C_CONTIGUOUS"]:
        arr = np.ascontiguousarray(arr)
    raw = memoryview(arr)
    nbytes = raw.nbytes

    # Simulate network latency and bandwidth throttling if enabled
    if _sim_latency_s > 0 or _sim_bw_mbps > 0:
        delay = _sim_latency_s
        if _sim_bw_mbps > 0:
            delay += nbytes / (_sim_bw_mbps * 125000.0)  # MB/s to B/s
        time.sleep(delay)

    # Send 8-byte length header followed by raw bytes
    header = struct.pack("!Q", nbytes)
    sock.sendall(header)
    sock.sendall(raw)

def recv(tensor, src):
    if src == _rank:
        return
    if src not in _recv_socks:
        raise RuntimeError(f"Rank {_rank} has no connection from src={src}")

    sock = _recv_socks[src]
    arr = tensor.detach().cpu().numpy()
    if not arr.flags["C_CONTIGUOUS"]:
        raise ValueError("Received tensor buffer must be C-contiguous")
    raw = memoryview(arr)
    expected_bytes = raw.nbytes

    # Read 8-byte length header
    header = b""
    while len(header) < 8:
        chunk = sock.recv(8 - len(header))
        if not chunk:
            raise ConnectionResetError(f"Socket closed while receiving from rank {src}")
        header += chunk
    payload_len = struct.unpack("!Q", header)[0]
    if payload_len != expected_bytes:
        raise ValueError(f"Size mismatch: expected {expected_bytes} bytes, received {payload_len} bytes")

    # Read into tensor memory directly
    received = 0
    while received < expected_bytes:
        n = sock.recv_into(raw[received:])
        if n == 0:
            raise ConnectionResetError(f"Socket closed during payload from rank {src}")
        received += n

def barrier():
    # Simple barrier: ring token exchange
    token = torch.zeros(1, dtype=torch.int32)
    next_rank = (_rank + 1) % _world_size
    prev_rank = (_rank - 1 + _world_size) % _world_size
    if _world_size > 1:
        if _rank == 0:
            send(token, dst=next_rank)
            recv(token, src=prev_rank)
        else:
            recv(token, src=prev_rank)
            send(token, dst=next_rank)

def destroy_process_group():
    global _listen_sock, _send_socks, _recv_socks
    for s in _send_socks.values():
        try:
            s.close()
        except Exception:
            pass
    for s in _recv_socks.values():
        try:
            s.close()
        except Exception:
            pass
    if _listen_sock:
        try:
            _listen_sock.close()
        except Exception:
            pass
    _send_socks.clear()
    _recv_socks.clear()

def is_available():
    return True
