import argparse
import os
import torch
import logging
import time
import copy
import math

try:
    if os.environ.get("USE_REAL_DIST", "0") == "1":
        from torch import distributed as dist
    else:
        import sim_dist as dist
except Exception:
    import sim_dist as dist

DEVICE = "cpu"
#TENSOR_SIZE = 4
send_time = 0.0
receive_time = 0.0

def init_process(master_ip, rank, world_size):
    if ":" not in master_ip:
        init_method = "tcp://" + master_ip + ":6585"
    else:
        init_method = "tcp://" + master_ip
    dist.init_process_group(backend="gloo",
                            init_method=init_method,
                            rank=rank,
                            world_size=world_size)

def main(rank, num_nodes, tensor_size):
    blk_size = tensor_size // num_nodes
    rank = dist.get_rank()
    inp = torch.full([tensor_size], rank + 1, dtype=torch.float16)

    bde_allreduce(rank, num_nodes, inp, 0, tensor_size, depth=0)

    expected_sum = num_nodes * (num_nodes + 1) // 2
    expected_tensor = torch.full_like(inp, fill_value = expected_sum)
    assert torch.allclose(expected_tensor, inp), f"wrong output by {torch.max(expected_tensor - inp)}"
    print(f"Rank {rank} has correct allreduct output of : {inp}") 

def bde_allreduce(rank, num_nodes, inp, l, r, depth):
    if depth == int(math.log2(num_nodes)):
        return

    mid = (l + r) // 2
    half_size = (r - l ) // 2
    recv_buffer = torch.zeros([half_size], dtype=inp.dtype)
    is_left = (rank // (2**depth)) % 2 == 0
    partner = rank + 2**depth if is_left else rank - 2**depth

    #1 . HALVING
    if is_left:
        dist.send(inp[mid:r], partner)
        dist.recv(recv_buffer, partner)
        inp[l:mid] += recv_buffer.clone()
        bde_allreduce(rank, num_nodes, inp, l, mid, depth + 1)
    else:
        dist.recv(recv_buffer, partner)
        dist.send(inp[l:mid], partner)
        inp[mid:r] += recv_buffer.clone()
        bde_allreduce(rank, num_nodes, inp, mid, r, depth + 1)

    # DOUBLING
    if is_left:
        dist.send(inp[l:mid], partner)
        dist.recv(recv_buffer, partner)
        inp[mid:r] = recv_buffer
    else:
        dist.recv(recv_buffer, partner)
        dist.send(inp[mid:r], partner)
        inp[l:mid] = recv_buffer
    return

# Implementation below has wrong errors and inefficiency such as:
#   1. haven't calculated end position during Doubling
#   2. notice how both Halving and Doulbing can be combined inside a recursive function, like in merge sort?
#   3. Confuse between buffer size (require halving) with (r - l)
#   4. wrong base case in halving(), should check depth, not l == r

#     halving(rank, num_nodes, inp, 0, len(tensor_size), depth=0)

#     doubling(rank, num_nodes, inp, blk_size, depth=0)
    
# def halving(rank, num_nodes, inp, l, r, depth):
#     if l == r:
#         return
#     assert (l + r) % 2 == 0 , "currently only support len(tensor_size) == exp(2)"
#     mid = int((l + r) / 2)
#     recv_buffer = torch.zeros([r - l], dtype = inp.dtype)
#     if (rank // 2**depth) % 2 == 0:
#         # left
#         dst = src = rank + 2**depth
#         dist.send(inp[mid: r], dst)
#         dist.recv(recv_buffer, src)
#         inp[l: mid] += recv_buffer # no need .clone() because won't use buffer again, it's recursively called new
#         halving(rank, num_nodes, inp, l, mid, depth + 1)
#     else:
#         # right
#         dst = src = rank - 2**depth
#         dist.recv(recv_buffer, src)
#         dist.send(inp[l: mid], dst)
#         inp[mid : r] += recv_buffer
#         halving(rank, num_nodes, inp, mid, r, depth + 1)
#     return

# def doubling(rank, num_nodes, inp, blk_size, depth):
#     if depth == int(math.log2(num_nodes)):
#         return

#     start = rank * blk_size
#     end = 0  #fnc(start, depth, blk_size)
#     recv_buffer = torch.zeros([end - start], dtype = inp.dtype)
#     if (rank // 2**depth) % 2 == 0:
#         #left
#         dst = src = rank + 2**depth
#         dist.send(inp[start: end], dst)
#         dist.recv(recv_buffer, src)
#         src_start = dst * blk_size
#         src_end = 0 # fnc(src_start, blk_size, depth)
#         inp[src_start: src_end] = recv_buffer
#         doubling(rank, num_nodes, inp, blk_size, depth + 1)
#     else:
#         #right
#         dst = src = rank - 2**depth
#         dist.recv(recv_buffer, src)
#         dist.send(inp[start:end], dst)
#         src_start = dst * blk_size
#         src_end = 0 # fnc(src_start, blk_size, depth)
#         inp[src_start: src_end] = recv_buffer
#         doubling(rank, num_nodes, inp, blk_size, depth + 1)
#     return

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--master-ip", "-m", required=True, type=str)
    parser.add_argument("--num-nodes", "-n", required=True, type=int)
    parser.add_argument("--rank", "-r", required=True, type=int)
    parser.add_argument("--tensor-size", "-t", required=True, type=int)


    args = parser.parse_args()
    #print("Rank",args.rank,"entered main")
    init_process(master_ip=args.master_ip,
                 rank=args.rank,
                 world_size=args.num_nodes)
    main(rank=args.rank, num_nodes=args.num_nodes, tensor_size=args.tensor_size)
