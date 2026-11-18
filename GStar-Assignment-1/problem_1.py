import torch
import torch.nn as nn
import math
import torch.nn.functional as F

class FlashAttention2Function(torch.autograd.Function):
    """
    A pure PyTorch implementation of the FlashAttention-2 forward pass.
    This version is a template for student implementation.
    """

    # @staticmethod
    # def forward(ctx, Q, K, V, is_causal = False):
    #     B, H, S_q, h_q = Q.shape
    #     _, _, S_k, h_k = V.shape 

    #     Q_TILES, K_TILES = 128, 128
    #     N_Q_TILES = math.ceil(S_q / Q_TILES)
    #     N_K_TILES = math.ceil(S_k / K_TILES)

    #     L_o = torch.zeros([B, H, S_q], device = Q.device, dtype = torch.float32) # lse
    #     O_o = torch.zeros_like(Q, dtype = Q.dtype) # attn output

    #     scale = 1 / math.sqrt(h_q)
    #     for b in range(B):
    #         for h in range(H):
    #             q_bh = Q[b, h, :, :]
    #             k_bh = K[b, h, :, :]
    #             v_bh = V[b, h, :, :]

    #             for i in range(N_Q_TILES):
    #                 start_q = i * Q_TILES
    #                 end_q = min((i + 1) * Q_TILES, S_q)
    #                 q_tile = q_bh[start_q: end_q, :]
                    
    #                 m_i = torch.full((end_q - start_q,), -float('inf'), device = Q.device, dtype = torch.float32)
    #                 l_i = torch.zeros((end_q - start_q,), device = Q.device, dtype = torch.float32) # sum exponentials
    #                 o_i = torch.zeros_like(q_tile, device = Q.device, dtype=Q.dtype) # [tq, hd]

    #                 for j in range(N_K_TILES):
    #                     start_k = j * K_TILES
    #                     end_k = min((j + 1) * K_TILES, S_k)
    #                     k_tile = k_bh[start_k : end_k, :]
    #                     v_tile = v_bh[start_k : end_k, :]

    #                     # calculate running stats
    #                     s_ij = (q_tile @ k_tile.transpose(-1, -2)).to(torch.float32) * scale
    #                     if is_causal:
    #                         q_idx = torch.arange(start_q, end_q, device = Q.device).unsqueeze(-1) # [tq, 1]
    #                         k_idx = torch.arange(start_k, end_k, device = Q.device).unsqueeze(0) # [1, tk]
    #                         s_ij = s_ij.masked_fill(k_idx > q_idx, -float('inf')) # satisfied should give use True
    #                     m_ij = torch.max(s_ij, dim = -1).values # torch.max returns (val, idx)
    #                     m_new = torch.maximum(m_i, m_ij)
    #                     # correct
    #                     scale_factor = torch.exp(m_i - m_new)
    #                     l_i *= scale_factor
    #                     o_i *= scale_factor.unsqueeze(-1)
    #                     # update
    #                     p_tile_ij = torch.exp(s_ij - m_new.unsqueeze(-1))
    #                     l_i += torch.sum(p_tile_ij, dim = -1)
    #                     o_i += p_tile_ij.to(Q.dtype) @ v_tile
    #                     m_i = m_new
    #                 normalized_li = torch.where(l_i > 0, 1.0 / l_i, 0)
    #                 L_o[b, h, start_q : end_q] = m_i + torch.log(l_i)
    #                 normalized_oi = o_i * normalized_li.unsqueeze(-1)
    #                 O_o[b, h, start_q: end_q, :] = normalized_oi 

    #     return O_o.to(Q.dtype), L_o

    @staticmethod
    def forward(ctx, Q, K, V, is_causal = False):
        Bq, Hq, Sq, D = Q.shape
        _ , _, Sh, _ = K.shape

        Q_TILE_SIZE = 256
        K_TILE_SIZE = 256

        N_Q_TILES = math.ceil(Sq / Q_TILE_SIZE)
        N_K_TILES = math.ceil(Sh / K_TILE_SIZE)

        scale = 1 / math.sqrt(D) # Python float

        O_o = torch.zeros_like(Q) # float16
        L_o = torch.zeros([Bq, Hq, Sq], device = Q.device, dtype = Q.dtype) # float16

        # for b in range(Bq):
        #     for h in range(Hq):
        #         Qbh = Q[b, h, :, :]
        #         Kbh = K[b, h, :, :]
        #         Vbh = V[b, h, :, :]

        # traverse across tiles of Q
        for q_idx in range(N_Q_TILES):
            q_start = q_idx * Q_TILE_SIZE
            q_end = min((q_idx + 1) * Q_TILE_SIZE, Sq)
            q_tile = Q[:, :, q_start: q_end, :]

            local_o = torch.zeros_like(q_tile, dtype=torch.float32) # float32
            local_m = torch.full([Bq, Hq, q_end - q_start], -float('inf'), dtype = torch.float32, device = Q.device) # on-chip with Q
            local_lse = torch.zeros([Bq, Hq, q_end - q_start], dtype = torch.float32, device = Q.device) # on-chip
            # for each q_tile, traverse across full KV dimensions to get final attn output for q_tile
            for k_idx in range(N_K_TILES):
                k_start = k_idx * K_TILE_SIZE
                k_end = min((k_idx + 1) * K_TILE_SIZE, Sh)
                k_tile = K[:, :, k_start: k_end, :]
                v_tile = V[:, :, k_start: k_end, :]
                # get causal attnn
                # s_qk = ((q_tile @ k_tile.T) * scale).to(torch.float32)  # multiplies 2 float16 matrices, but accumulate to a float32 output
                # this operation means: 
                #   1. q_tile @ k_tile.T may use Tensor Cores with FP32 internal accumulatoin
                #   2. the matric multiplication results is rounded back to FP16
                #   3. Multiplciation by scale (float16 x Python float) gives us a FP16 output
                #   4. then scale fp16 to fp32 => This sequence converts the already rounded fp16 to fp32, it cannot recover lost precision
                # Another approahc
                s_qk = (q_tile @ k_tile.transpose(-1, -2)).float() * scale # .float() performs scaling in FP32 first.
                # Before: fp16 x fp16 = fp16 x Python scale -> fp16 -(rounds)-> fp32 ! lost information with low precision scale multiplication
                # Now: fp16 x fp16 = float16 -(rounds)-> fp32 x Python scale -> fp32 => keeps information with high precision scale multiplication
                if is_causal:
                    causal_q = torch.arange(q_start, q_end, device = Q.device).unsqueeze(-1)
                    causal_k = torch.arange(k_start, k_end, device = Q.device).unsqueeze(0)
                    s_qk = s_qk.masked_fill(causal_q < causal_k, -float('inf'))  # float-16, if the indices don't change the dtype
                # update local stats
                new_m = torch.max(s_qk, dim = -1).values
                new_m = torch.maximum(new_m, local_m)
                scale_factor = torch.exp(local_m - new_m) # [s_q,]
                # correct outputs
                local_lse *= scale_factor # [sq]
                local_o *= scale_factor.unsqueeze(-1) # [s_q, h_q] * [s_q,1]
                # add current tiles
                P_tile = torch.exp(s_qk - new_m.unsqueeze(-1)) # [sq, hk], probabilities
                local_lse += torch.sum(P_tile, dim=-1) # [sq, sk] - [sq, 1]
                local_o += P_tile @ v_tile.to(torch.float32) # [sq, sk] @ [sk, hk] = [sq, hk]
                local_m = new_m
            # set q_tile answer to Ouput
            reciprocal_lse = torch.where(local_lse > 0, 1 / local_lse, 0)
            normalized_out = local_o * reciprocal_lse.unsqueeze(-1)
            O_o[:, :, q_start : q_end, :] = normalized_out.to(Q.dtype)
            L_o[:, :, q_start : q_end] = (torch.log(local_lse) + local_m).to(Q.dtype)

        return O_o, L_o
    
    @staticmethod
    def backward(ctx, grad_out, grad_L):
        raise NotImplementedError("Backward pass not yet implemented for FlashAttention2Function")

def benchmark_fn(fn, *args, warmup=1, iters=3, **kwargs):
    # Warmup runs to initialize CUDA kernels and caches
    for _ in range(warmup):
        fn(*args, **kwargs)
    torch.cuda.synchronize()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    for _ in range(iters):
        fn(*args, **kwargs)
    end_event.record()
    torch.cuda.synchronize()

    return start_event.elapsed_time(end_event) / iters  # average time in milliseconds


def naive_attention(q, k, v, is_causal=False):
    s = (q.float() @ k.float().transpose(-1, -2)) / math.sqrt(q.shape[-1])
    if is_causal:
        q_pos = torch.arange(0, q.shape[-2], device=q.device)
        k_pos = torch.arange(0, k.shape[-2], device=k.device)
        s.masked_fill_(q_pos[:, None] < k_pos[None, :], -float('inf'))
    p = torch.softmax(s, dim=-1).to(v.dtype)
    return p @ v


if __name__ == "__main__":
    causal = False
    Q = torch.randn(1, 2, 1024, 32, dtype=torch.float16, device = 'cuda')
    K = torch.randn(1, 2, 1024, 32, dtype=torch.float16, device = 'cuda')
    V = torch.randn(1, 2, 1024, 32, dtype=torch.float16, device = 'cuda')

    # Accuracy verification
    O_ref = F.scaled_dot_product_attention(Q, K, V, is_causal=causal)

    def custom_attention(Q, K, V, causal=True):
        return FlashAttention2Function.apply(Q, K, V, causal)
    compiled_attention = torch.compile(custom_attention, mode = "default")
    with torch.no_grad():
        O, L = compiled_attention(Q, K, V, causal)
        torch.cuda.synchronize()

    scores = (Q.float() @ K.float().transpose(-1, -2)) / math.sqrt(Q.shape[-1])
    if causal:
        q_pos = torch.arange(0, Q.shape[-2], device=Q.device)
        k_pos = torch.arange(0, K.shape[-2], device=K.device)
        scores.masked_fill_(q_pos[:, None] < k_pos[None, :], -float('inf'))
    L_ref = torch.logsumexp(scores, dim=-1)

    print("Accuracy check:")
    print("  Output max error:", (O.float() - O_ref.float()).abs().max().item())
    print("  LSE max error:   ", (L.float() - L_ref.float()).abs().max().item())
    print("  O shape & dtype: ", tuple(O.shape), O.dtype)
    print("  L shape & dtype: ", tuple(L.shape), L.dtype)

    # Performance timing comparison
    print("\nTiming comparison (average over 3 iterations):")
    t_sdpa = benchmark_fn(F.scaled_dot_product_attention, Q, K, V, is_causal=causal)
    t_naive = benchmark_fn(naive_attention, Q, K, V, is_causal=causal)
    t_custom = benchmark_fn(compiled_attention, Q, K, V, causal)

    print(f"  PyTorch SDPA (native kernel): {t_sdpa:.4f} ms")
    print(f"  Standard Naive Attention:    {t_naive:.4f} ms")
    print(f"  Custom FlashAttention-2:     {t_custom:.4f} ms")

# Output from running python problem1.py
# Before switching online stats to float32
# Output max error 0.00048828125
# LSE max error  0.004456520080566406
# O: (1, 2, 1024, 32) torch.float16
# L: (1, 2, 1024) torch.float16
# 
# Afterefore switching online stats to float32
# Output max error 0.0001220703125
# LSE max error  0.001953125
# O: (1, 2, 1024, 32) torch.float16
# L: (1, 2, 1024) torch.float16

# => Output max error from 4.88x10^-4 to 1.22x10^-4 and LSE max error from 4.45x10^-3 to 1.95x10^-3

# After s_qk precision scaling adjustment
#   Output max error: 0.000732421875
#   LSE max error:    0.001987457275390625
#   O shape & dtype:  (1, 2, 1024, 32) torch.float16
#   L shape & dtype:  (1, 2, 1024) torch.float16
# => Not make quite a difference
# Timing comparison (average over 3 iterations):
#   PyTorch SDPA (native kernel): 0.1417 ms
#   Standard Naive Attention:    0.9721 ms
#   Custom FlashAttention-2:     39.1561 ms

# After parallel B, H dimension with batching
#   Output max error: 0.00048828125
#   LSE max error:    0.0021562576293945312
#   O shape & dtype:  (1, 2, 1024, 32) torch.float16
#   L shape & dtype:  (1, 2, 1024) torch.float16
# Timing comparison (average over 3 iterations):
#   PyTorch SDPA (native kernel): 0.1939 ms
#   Standard Naive Attention:    0.6854 ms
#   Custom FlashAttention-2:     22.1460 ms

# After torch.compile()
# Accuracy check:
#   Output max error: 0.000244140625
#   LSE max error:    0.002002239227294922
#   O shape & dtype:  (1, 2, 1024, 32) torch.float16
#   L shape & dtype:  (1, 2, 1024) torch.float16

# Timing comparison (average over 3 iterations):
#   PyTorch SDPA (native kernel): 0.1256 ms
#   Standard Naive Attention:    0.8919 ms
#   Custom FlashAttention-2:     6.9041 ms

# => Much faster time! 22ms down to 6.9ms