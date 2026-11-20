"""
Interactive Practice Environment for Collective AllReduce Algorithms
====================================================================
Practice implementing:
1. Ring AllReduce (Reduce-Scatter + All-Gather)
2. Recursive Halving and Doubling (BDE)

Run directly with:
    python local_runner.py --algo ring --nodes 4 --tensor-size 1024
or run automated test suite:
    python practice_allreduce.py
"""

import os
import sys
import time
import argparse
import torch

try:
    if os.environ.get("USE_REAL_DIST", "0") == "1":
        from torch import distributed as dist
    else:
        import sim_dist as dist
except Exception:
    import sim_dist as dist


def ring_allreduce(tensor, rank, world_size):
    """
    Implements Ring AllReduce on 1D tensor:
    - Step 1: Reduce-Scatter (world_size - 1 steps)
      Each rank sends chunk (rank + i) % P to (rank - 1) % P
      and receives chunk from (rank + 1) % P, adding to its buffer.
    - Step 2: All-Gather (world_size - 1 steps)
      Each rank sends the fully reduced chunk to (rank + 1) % P
      and receives into its missing chunk from (rank - 1) % P.
    """
    tensor_size = tensor.numel()
    chunk_size = tensor_size // world_size
    chunks = list(tensor.split(chunk_size))
    
    recv_buffer = torch.zeros(chunk_size, dtype=tensor.dtype)
    
    # -------------------------------------------------------------
    # Phase 1: Reduce-Scatter (P - 1 steps)
    # -------------------------------------------------------------
    for i in range(1, world_size):
        send_idx = (rank + i) % world_size
        recv_idx = (rank + i + 1) % world_size
        dst = (rank + world_size - 1) % world_size
        src = (rank + 1) % world_size
        
        # Deadlock prevention in synchronous send/recv:
        # Alternating even/odd ranks
        if rank % 2 == 0:
            dist.send(chunks[send_idx], dst=dst)
            dist.recv(recv_buffer, src=src)
        else:
            dist.recv(recv_buffer, src=src)
            dist.send(chunks[send_idx], dst=dst)
            
        chunks[recv_idx] += recv_buffer

    # -------------------------------------------------------------
    # Phase 2: All-Gather (P - 1 steps)
    # -------------------------------------------------------------
    for i in range(1, world_size):
        send_idx = (rank + 1 - i + world_size) % world_size
        recv_idx = (rank - i + world_size) % world_size
        dst = (rank + 1) % world_size
        src = (rank + world_size - 1) % world_size
        
        if rank % 2 == 0:
            dist.send(chunks[send_idx], dst=dst)
            dist.recv(chunks[recv_idx], src=src)
        else:
            dist.recv(chunks[recv_idx], src=src)
            dist.send(chunks[send_idx], dst=dst)

    # Reassemble tensor
    tensor.copy_(torch.cat(chunks))
    return tensor


def recursive_hd_allreduce(tensor, rank, world_size):
    """
    Implements Recursive Halving and Doubling (BDE - Bidirectional Exchange) AllReduce.
    Assumes world_size is a power of 2.
    - Halving phase: log2(P) steps reducing message size by half each step.
    - Doubling phase: log2(P) steps expanding message size by double each step.
    """
    tensor_len = len(tensor)
    
    def reduce_scatter(x, t_left, t_right, r_left, r_right):
        if r_left == r_right:
            return
        r_size = r_right - r_left + 1
        t_mid = (t_left + t_right) // 2
        r_mid = (r_left + r_right) // 2
        
        partner = int(rank + r_size // 2 if rank <= r_mid else rank - r_size // 2)
        
        if rank <= r_mid:
            # Send top half, receive bottom half
            dist.send(x[t_mid+1:t_right+1], dst=partner)
            recv_buf = torch.zeros(t_mid - t_left + 1, dtype=x.dtype)
            dist.recv(recv_buf, src=partner)
            x[t_left:t_mid+1] += recv_buf
            reduce_scatter(x, t_left, t_mid, r_left, r_mid)
        else:
            # Receive top half, send bottom half
            recv_buf = torch.zeros(t_right - t_mid, dtype=x.dtype)
            dist.recv(recv_buf, src=partner)
            x[t_mid+1:t_right+1] += recv_buf
            dist.send(x[t_left:t_mid+1], dst=partner)
            reduce_scatter(x, t_mid+1, t_right, r_mid+1, r_right)

    def all_gather(x, t_left, t_right, r_left, r_right):
        if r_left == r_right:
            return
        r_size = r_right - r_left + 1
        t_mid = (t_left + t_right) // 2
        r_mid = (r_left + r_right) // 2
        
        partner = int(rank + r_size // 2 if rank <= r_mid else rank - r_size // 2)
        
        if rank <= r_mid:
            all_gather(x, t_left, t_mid, r_left, r_mid)
            dist.send(x[t_left:t_mid+1], dst=partner)
            recv_buf = x[t_mid+1:t_right+1]
            dist.recv(recv_buf, src=partner)
        else:
            all_gather(x, t_mid+1, t_right, r_mid+1, r_right)
            recv_buf = x[t_left:t_mid+1]
            dist.recv(recv_buf, src=partner)
            dist.send(x[t_mid+1:t_right+1], dst=partner)

    reduce_scatter(tensor, 0, tensor_len - 1, 0, world_size - 1)
    all_gather(tensor, 0, tensor_len - 1, 0, world_size - 1)
    return tensor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--master-ip", "-m", default="127.0.0.1:16500", type=str)
    parser.add_argument("--num-nodes", "-n", default=4, type=int)
    parser.add_argument("--rank", "-r", default=0, type=int)
    parser.add_argument("--tensor-size", "-t", default=1024, type=int)
    parser.add_argument("--algo", "-a", choices=["ring", "bde"], default="ring")
    args = parser.parse_args()

    # Initialize distributed process
    init_method = args.master_ip if ":" in args.master_ip else f"{args.master_ip}:6585"
    if not init_method.startswith("tcp://"):
        init_method = "tcp://" + init_method

    dist.init_process_group(
        backend="gloo",
        init_method=init_method,
        rank=args.rank,
        world_size=args.num_nodes
    )

    # Initial deterministic tensor: rank + 1 everywhere
    # Total sum across ranks 0..P-1 will be sum(1..P) = P*(P+1)/2
    t = torch.ones(args.tensor_size, dtype=torch.float32) * (args.rank + 1)
    expected_sum = args.num_nodes * (args.num_nodes + 1) / 2.0

    t0 = time.time()
    if args.algo == "ring":
        ring_allreduce(t, args.rank, args.num_nodes)
    else:
        recursive_hd_allreduce(t, args.rank, args.num_nodes)
    t1 = time.time()

    is_correct = torch.allclose(t, torch.tensor(expected_sum, dtype=torch.float32))
    print(f"Rank {args.rank}: {args.algo.upper()} AllReduce done in {t1 - t0:.6f}s | Correct: {is_correct}")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
