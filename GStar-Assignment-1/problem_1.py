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

        scale = 1 / math.sqrt(Hq)

        O_o = torch.zeros_like(Q) # float16
        L_o = torch.zeros([Bq, Hq, Sq], device = Q.device, dtype = Q.dtype) # float16

        for b in range(Bq):
            for h in range(Hq):
                Qbh = Q[b, h, :, :]
                Kbh = K[b, h, :, :]
                Vbh = V[b, h, :, :]

                # traverse across tiles of Q
                for q_idx in range(N_Q_TILES):
                    q_start = q_idx * Q_TILE_SIZE
                    q_end = min((q_idx + 1) * Q_TILE_SIZE, Sq)
                    q_tile = Qbh[q_start: q_end, :]

                    local_o = torch.zeros_like(q_tile) # float16
                    local_m = torch.full((q_tile.shape[0],), -float('inf'), dtype = Q.dtype, device = Q.device) # on-chip with Q
                    local_lse = torch.zeros(q_tile.shape[0], dtype = Q.dtype, device = Q.device) # on-chip
                    # for each q_tile, traverse across full KV dimensions to get final attn output for q_tile
                    for k_idx in range(N_K_TILES):
                        k_start = k_idx * K_TILE_SIZE
                        k_end = min((k_idx + 1) * K_TILE_SIZE, Sh)
                        k_tile = Kbh[k_start: k_end, :]
                        v_tile = Vbh[k_start: k_end, :]
                        # get causal attnn
                        s_qk = (q_tile @ k_tile.T) * scale  # float16
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
                        local_o += P_tile @ v_tile # [sq, sk] @ [sk, hk] = [sq, hk]
                        local_m = new_m
                    # set q_tile answer to Ouput
                    reciprocal_lse = torch.where(local_lse > 0, 1 / local_lse, 0)
                    O_o[b, h, q_start : q_end, :] = local_o * reciprocal_lse.unsqueeze(-1)
                    L_o[b, h, q_start : q_end] = torch.log(local_lse) + local_m

        return O_o, L_o
    
    @staticmethod
    def backward(ctx, grad_out, grad_L):
        raise NotImplementedError("Backward pass not yet implemented for FlashAttention2Function")

if __name__ == "__main__":
    causal = False
    Q = torch.randn(1, 2, 1024, 32, dtype=torch.float16, device = 'cuda')
    K = torch.randn(1, 2, 1024, 32, dtype=torch.float16, device = 'cuda')
    V = torch.randn(1, 2, 1024, 32, dtype=torch.float16, device = 'cuda')

    O_ref = F.scaled_dot_product_attention(Q, K,V , is_causal=causal)
    O, L = FlashAttention2Function.apply(Q, K, V, causal)

    scores = (Q.float() @ K.float().transpose(-1, -2)) / math.sqrt(Q.shape[-1])
    if causal:
        q_pos = torch.arange(0, Q.shape[-2])
        k_pos = torch.arange(0, K.shape[-2])
        scores.masked_fill_(q_pos[:, None] < k_pos[None, :], -float('inf'))
    L_ref = torch.logsumexp(scores, dim=-1)
    print("Output max error", (O.float() - O_ref.float()).abs().max().item())
    print("LSE max error ", (L.float() - L_ref.float()).abs().max().item())
    print("O:", tuple(O.shape), O.dtype)
    print("L:", tuple(L.shape), L.dtype)

# Output from running python problem1.py
# Output max error 3.1571044921875
# LSE max error  16.199954986572266
# O: (1, 2, 1024, 32) torch.float16
# L: (1, 2, 1024) torch.float16
