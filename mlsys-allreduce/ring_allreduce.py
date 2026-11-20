import argparse
import os
import torch
import logging
import time

try:
    if os.environ.get("USE_REAL_DIST", "0") == "1":
        from torch import distributed as dist
    else:
        import sim_dist as dist
except Exception:
    import sim_dist as dist

DEVICE = "cpu"
#TENSOR_SIZE = 1024

def init_process(master_ip, rank, world_size):
    if ":" not in master_ip:
        init_method = "tcp://" + master_ip + ":6585"
    else:
        init_method = "tcp://" + master_ip
    dist.init_process_group(backend="gloo",
                            init_method=init_method,
                            rank=rank,
                            world_size=world_size)

def main(tensor_size):
    # Get world size and rank
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    total_time = 0
    # Create a random tensor
    x = torch.full([tensor_size], rank + 1)
    # Create send and receive buffers
    x_blks = list(x.split(int(tensor_size / world_size)))
    recv_buffer = torch.zeros([int(tensor_size/ world_size)], dtype = x.dtype, device = x.device)
    print("Rank", rank, "start RS loop")
    # Reduce-scatter loop
    for i in range(world_size - 1):
        dst = (rank + world_size - 1) % world_size 
        src = (rank  + 1) % world_size
        chunk_id = (rank + i) % world_size
        s = time.time()
        if rank % 2 == 0:
            # Send a tensor to the previous machine
            dist.send(x_blks[chunk_id], dst)        
            # Receive a tensor from the next machine
            dist.recv(recv_buffer, src)
        else:
            # Receive a tensor from the next machine
            dist.recv(recv_buffer, src)
            # Send a tensor to the previous machine
            dist.send(x_blks[chunk_id], dst)
        e = time.time()
        total_time += (e - s)
        # Accumulate value in t. At the end of the for loop, t will hold the reduced value
        x_blks[(chunk_id + 1) % world_size] += recv_buffer.clone()
        print("Rank", rank, " updated chunk_id: ", (chunk_id + 1) % world_size)
        recv_buffer = torch.zeros_like(recv_buffer)
    print("Rank", rank, "reduce scatter time:", total_time)
    # All-gather loop
    for i in range(world_size - 1):
        dst = (rank + world_size - 1) % world_size 
        src = (rank  + 1) % world_size
        chunk_id = (rank + i - 1 + world_size) % world_size
        s = time.time()
        if rank % 2 == 0:
            # Send a tensor to the next machine
            dist.send(x_blks[chunk_id], dst)
            # Receive a tensor from the previous machine
            dist.recv(recv_buffer, src)
        else:
            # Receive a tensor from the previous machine
            dist.recv(recv_buffer, src)
            # Send a tensor to the next machine
            dist.send(x_blks[chunk_id], dst)
        e = time.time()
        total_time += (e - s)
        x_blks[(chunk_id + 1) % world_size] = recv_buffer.clone()
        recv_buffer = torch.zeros_like(recv_buffer)
    print("Rank", rank, "finished allreduce in ", total_time, " seconds")
    out = torch.cat(x_blks)

    expected_sum = world_size * (world_size + 1) / 2.0
    expected_tensor = torch.full_like(out, fill_value = expected_sum)
    assert torch.allclose(expected_tensor, out), f"Rank {rank} got wrong sum"
    print(f"Rank {rank} is correct")
    #print(torch.cat(t))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--master-ip", "-m", required=True, type=str)
    parser.add_argument("--num-nodes", "-n", required=True, type=int)
    parser.add_argument("--rank", "-r", required=True, type=int)
    parser.add_argument("--tensor-size", "-t", required=True, type=int)

    args = parser.parse_args()
    init_process(master_ip=args.master_ip,
                 rank=args.rank,
                 world_size=args.num_nodes)
    main(tensor_size=args.tensor_size)
