import torch
N = 13
world_size = 5
zero_buffer = torch.zeros([N], dtype = torch.float32, device = "cuda")
recv_buffer = list(zero_buffer.split(int( N / world_size))
print(recv_buffer)
