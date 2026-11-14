import torch
import torch.nn as nn
import math

class FlashAttention2Function(torch.autograd.Function):
    """
    A pure PyTorch implementation of the FlashAttention-2 forward pass.
    This version is a template for student implementation.
    """

    @staticmethod
    def forward(ctx, Q, K, V, is_causal = False):
        B, H, S_q, h_q = Q.shape
        _, _, S_k, h_k = V.shape 

        Q_TILES, K_TILES = 128, 128
        N_Q_TILES = math.ceil(S_q / Q_TILES)
        N_K_TILES = math.ceil(S_k / K_TILES)

        L_o = torch.zeros([B, H, S_q], device = Q.device, dtype = torch.float32) # lse
        O_o = torch.zeros_like(Q, dtype = Q.dtype) # attn output

        scale = 1 / math.sqrt(h_q)
        for b in range(B):
            for h in range(H):
                q_bh = Q[b, h, :, :]
                k_bh = K[b, h, :, :]
                v_bh = V[b, h, :, :]

                for i in range(N_Q_TILES):
                    start_q = i * Q_TILES
                    end_q = min((i + 1) * Q_TILES, S_q)
                    q_tile = q_bh[start_q: end_q, :]
                    
                    m_i = torch.full((end_q - start_q,), -float('inf'), device = Q.device, dtype = torch.float32)
                    l_i = torch.zeros((end_q - start_q,), device = Q.device, dtype = torch.float32) # sum exponentials
                    o_i = torch.zeros_like(q_tile, device = Q.device, dtype=Q.dtype) # [tq, hd]

                    for j in range(N_K_TILES):
                        start_k = j * K_TILES
                        end_k = min((j + 1) * K_TILES, S_k)
                        k_tile = k_bh[start_k : end_k, :]
                        v_tile = v_bh[start_k : end_k, :]

                        # calculate running stats
                        s_ij = (q_tile @ k_tile.transpose(-1, -2)).to(torch.float32) * scale
                        if is_causal:
                            q_idx = torch.arange(start_q, end_q, device = Q.device).unsqueeze(-1) # [tq, 1]
                            k_idx = torch.arange(start_k, end_k, device = Q.device).unsqueeze(0) # [1, tk]
                            s_ij = s_ij.masked_fill(k_idx > q_idx, -float('inf')) # satisfied should give use True
                        m_ij = torch.max(s_ij, dim = -1).values # torch.max returns (val, idx)
                        m_new = torch.maximum(m_i, m_ij)
                        # correct
                        scale_factor = torch.exp(m_i - m_new)
                        l_i *= scale_factor
                        o_i *= scale_factor.unsqueeze(-1)
                        # update
                        p_tile_ij = torch.exp(s_ij - m_new.unsqueeze(-1))
                        l_i += torch.sum(p_tile_ij, dim = -1)
                        o_i += p_tile_ij.to(Q.dtype) @ v_tile
                        m_i = m_new
                    normalized_li = torch.where(l_i > 0, 1.0 / l_i, 0)
                    L_o[b, h, start_q : end_q] = m_i + torch.log(l_i)
                    normalized_oi = o_i * normalized_li.unsqueeze(-1)
                    O_o[b, h, start_q: end_q, :] = normalized_oi 

        return O_o.to(Q.dtype), L_o


    # @staticmethod
    # def forward(ctx, Q, K, V, is_causal = False):
    #     B, H, N_Q, D_H = Q.shape 
    #     _, _, N_K, _ = K.shape 

    #     # define tile size
    #     Q_TILE_SIZE = 128
    #     K_TILE_SIZE = 128 

    #     N_Q_TILES = math.ceil(N_Q / Q_TILE_SIZE)
    #     N_K_TILES = math.ceil(N_K / K_TILE_SIZE)

    #     scale = 1.0 / math.sqrt(D_H)

    #     O_final = torch.zeros_like(Q, dtype=Q.dtype)
    #     L_final = torch.zeros((B, H, N_Q), device=Q.device, dtype=torch.float32)

    #     for b in range(B):
    #         for h in range(H):
    #             q_bh = Q[b, h, :, :]
    #             k_bh = K[b, h, :, :]
    #             v_bh = V[b, h, :, :]

    #             for i in range(N_Q_TILES):
    #                 q_start = i * Q_TILE_SIZE 
    #                 q_end = min((i + 1) * Q_TILE_SIZE, N_Q)
    #                 Q_tile = q_bh[q_start: q_end, :]

    #                 # accumulate in float32 for numerical safety (inputs may be fp16)
    #                 o_i = torch.zeros_like(Q_tile, dtype=torch.float32)
    #                 l_i = torch.zeros(q_end - q_start, device=Q.device, dtype=torch.float32)
    #                 m_i = torch.full((q_end - q_start,), -float('inf'), device=Q.device, dtype=torch.float32)

    #                 for j in range(N_K_TILES):
    #                     k_start = j * K_TILE_SIZE 
    #                     k_end = min((j + 1)* K_TILE_SIZE, N_K)
    #                     K_tile = k_bh[k_start: k_end, :]
    #                     V_tile = v_bh[k_start: k_end, :]

    #                     S_ij = (Q_tile @ K_tile.transpose(-2, -1)) * scale 
    #                     if is_causal:
    #                         q_idx = torch.arange(q_start, q_end, device=Q.device).unsqueeze(-1)
    #                         k_idx = torch.arange(k_start, k_end, device=Q.device).unsqueeze(0)
    #                         S_ij = S_ij.masked_fill(k_idx > q_idx, -float('inf'))
    #                     # torch.max returns (values, indices)
    #                     m_ij = torch.max(S_ij, dim=-1).values.to(torch.float32)
    #                     m_new = torch.maximum(m_i, m_ij)
    #                     # rescale previous accumulators
    #                     scale_factor = torch.exp(m_i - m_new) # (Q_t, )
    #                     l_i = l_i * scale_factor 
    #                     o_i = o_i * scale_factor.unsqueeze(-1)
    #                     # current tile contribution
    #                     P_tilde_ij = torch.exp(S_ij.to(torch.float32) - m_new.unsqueeze(-1))
    #                     l_i = l_i + torch.sum(P_tilde_ij, dim=-1)
    #                     o_i = o_i + (P_tilde_ij @ V_tile.to(torch.float32))
    #                     m_i = m_new

    #                 # normalize this query tile and write results back
    #                 l_i_reciprocal = torch.where(l_i > 0, 1.0 / l_i, 0)
    #                 o_i_normalized = o_i * l_i_reciprocal.unsqueeze(-1)

    #                 L_tile = m_i + torch.log(l_i) # log-sum-exp

    #                 O_final[b, h, q_start:q_end, :] = o_i_normalized.to(Q.dtype)
    #                 L_final[b, h, q_start:q_end] = L_tile
    
    #     O_final = O_final.to(Q.dtype)

    #     ctx.save_for_backward(Q, K, V, O_final, L_final)
    #     ctx.is_causal = is_causal
    #     return O_final, L_final
        

    # @staticmethod
    # def forward_fp32(Q, K, V, is_causal=False):
    #     # Get dimensions from input tensors following the (B, H, N, D) convention
    #     B, H, N_Q, D_H = Q.shape # N_Q is the number of query tokens, D_H is the hidden dimension
    #     _, _, N_K, _ = K.shape # N_K is the number of key tokens

    #     # Define tile sizes
    #     Q_TILE_SIZE = 128
    #     K_TILE_SIZE = 128
        
    #     N_Q_tiles = math.ceil(N_Q / Q_TILE_SIZE)
    #     N_K_tiles = math.ceil(N_K / K_TILE_SIZE)

    #     # Initialize final output tensors
    #     O_final = torch.zeros_like(Q, dtype=Q.dtype)
    #     L_final = torch.zeros((B, H, N_Q), device=Q.device, dtype=torch.float32)
        
    #     scale = 1.0 / math.sqrt(D_H)

    #     # Main loops: Iterate over each batch and head
    #     for b in range(B):
    #         for h in range(H):
    #             Q_bh = Q[b, h, :, :]
    #             K_bh = K[b, h, :, :]
    #             V_bh = V[b, h, :, :]

    #             # Loop over query tiles
    #             for i in range(N_Q_tiles):
    #                 q_start = i * Q_TILE_SIZE
    #                 q_end = min((i + 1) * Q_TILE_SIZE, N_Q)
    #                 Q_tile = Q_bh[q_start:q_end, :]

    #                 # Initialize accumulators for this query tile
    #                 o_i = torch.zeros_like(Q_tile, dtype=torch.float32) #running weighted output - the actual attention output.
    #                 l_i = torch.zeros(q_end - q_start, device=Q.device, dtype=torch.float32) #running sum of exponentials (for narmalization)
    #                 m_i = torch.full((q_end - q_start,), -float('inf'), device=Q.device, dtype=torch.float32) #max value for trick: subtracting the max to prevent exp(x) from blowing up

    #                 # Q [Nq, d] @ K.T [Nk, d] = X [Nq, Nk] -> Softmax(X, dim=-1) = A [Nq, Nk] -> A @ V [Nk, d] = O [Nq, d]
    #                 # Q [Tq, d] @ K1.T [Tk1, d] .. K2.T [Tk2, d] .. K3.T[Tk3, d] (sum(Tk_i) = Nk)
    #                 # X1 [Tq, Tk1] -> [Softmax(X1, dim=-1)] = A1 [Tq, Tk1] @ V1[Tk1, d] = O1[Tq, d] with m1 = max(X1, dim=-1) and l1 = log.sum(exp(X1), dim=-1)
    #                 # correction: fixing current m1, l1 then adding m2, l2 (represent traversing along the K's dimension in X) 
    #                 # Inner loop over key/value tiles
    #                 for j in range(N_K_tiles):
    #                     k_start = j * K_TILE_SIZE
    #                     k_end = min( (j+1)*K_TILE_SIZE, N_K)
    #                     K_tile, V_tile = K_bh[k_start:k_end, :], V_bh[k_start:k_end, :]
    #                     S_ij = (Q_tile @ K_tile.transpose(-1, -2)) * scale 
    #                     if is_causal:
    #                         q_idx = torch.arange(q_start, q_end, device = Q.device).unsqueeze(-1)
    #                         k_idx = torch.arange(k_start, k_end, device = Q.device).unsqueeze(0)
    #                         S_ij = S_ij.masked_fill(k_idx > q_idx, -float('inf'))
    #                     # torch.max returns a named tuple (values, indices)
    #                     m_ij = torch.max(S_ij, dim = -1).values.to(torch.float32)
    #                     max_new = torch.maximum(m_i, m_ij)
    #                     scale_factor = torch.exp(m_i - max_new) 
    #                     l_i = l_i * scale_factor  # correction
    #                     o_i = o_i * scale_factor.unsqueeze(-1).to(o_i.dtype)  # correction
    #                     P_tilde_ij = torch.exp(S_ij.to(torch.float32) - max_new.unsqueeze(-1)) # safe softmax
    #                     l_i = l_i + torch.sum(P_tilde_ij, dim=-1)
    #                     o_i = o_i + (P_tilde_ij @ V_tile.to(torch.float32)).to(o_i.dtype)
    #                     m_i = max_new
    #                     # --- END OF STUDENT IMPLEMENTATION ---

    #                 # After iterating through all key tiles, normalize the output
    #                 # This part is provided for you. It handles the final division safely.
    #                 l_i_reciprocal = torch.where(l_i > 0, 1.0 / l_i, 0) #
    #                 o_i_normalized = o_i * l_i_reciprocal.unsqueeze(-1)
                    
    #                 L_tile = m_i + torch.log(l_i)
                    
    #                 # Write results for this tile back to the final output tensors
    #                 O_final[b, h, q_start:q_end, :] = o_i_normalized
    #                 L_final[b, h, q_start:q_end] = L_tile
        
    #     O_final = O_final.to(Q.dtype)

    #     # ctx.save_for_backward(Q, K, V, O_final, L_final)
    #     # ctx.is_causal = is_causal
 
    #     return O_final, L_final
    
    @staticmethod
    def backward(ctx, grad_out, grad_L):
        raise NotImplementedError("Backward pass not yet implemented for FlashAttention2Function")

if __name__ == "__main__":
    Q = torch.randn(1, 2, 1024, 32, dtype=torch.float16)
    K = torch.randn(1, 2, 1024, 32, dtype=torch.float16)
    V = torch.randn(1, 2, 1024, 32, dtype=torch.float16)

    O, L = FlashAttention2Function.apply(Q, K, V, False)
    print("O:", tuple(O.shape), O.dtype)
    print("L:", tuple(L.shape), L.dtype)
