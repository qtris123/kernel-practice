import torch
import triton
import triton.language as tl
import math
import torch.cuda.nvtx as nvtx

# @triton.jit
# def _flash_attention_forward_causal_kernel(
#     # Pointers to Tensors
#     Q_ptr, K_ptr, V_ptr, O_ptr,
#     # Stride information for tensors
#     q_stride_b, q_stride_h, q_stride_s,
#     k_stride_b, k_stride_h, k_stride_s,
#     v_stride_b, v_stride_h, v_stride_s,
#     # Kernel parameters
#     softmax_scale,
#     SEQ_LEN,
#     N_HEADS,
#     # Constexpr tile sizes
#     HEAD_DIM: tl.constexpr,
#     BLOCK_M: tl.constexpr,
#     BLOCK_N: tl.constexpr,
# ):
#     """
#     Triton kernel for the forward pass of causal FlashAttention.
#     This is a template for student implementation.
#     """
#     # 1. Identify the block of queries and the batch/head to be processed.
#     q_block_idx = tl.program_id(axis=0)
#     batch_head_idx = tl.program_id(axis=1)
    
#     batch_idx = batch_head_idx // N_HEADS
#     head_idx = batch_head_idx % N_HEADS

#     # 2. Initialize accumulators in SRAM.
#     m_i = tl.full([BLOCK_M], -float('inf'), dtype=tl.float32)
#     l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
#     acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

#     # 3. Load the block of queries (Q_i).
#     q_offsets = (q_block_idx * BLOCK_M + tl.arange(0, BLOCK_M))
#     q_ptrs = Q_ptr + batch_idx * q_stride_b + head_idx * q_stride_h + \
#              (q_offsets[:, None] * q_stride_s + tl.arange(0, HEAD_DIM)[None, :])
#     q_block = tl.load(q_ptrs, mask=q_offsets[:, None] < SEQ_LEN, other=0.0)
    
#     # PyTorch softmax is exp(x), Triton is exp2(x * log2(e)), log2(e) is approx 1.44269504
#     qk_scale = softmax_scale * 1.44269504

#     # --- Phase 1: Accumulate in Off-Diagonal Blocks (No Masking) ---
#     # Process key/value blocks that are strictly in the past (q_idx > k_idx).
#     for start_n in range(0, q_block_idx * BLOCK_M, BLOCK_N):
#         # --- STUDENT IMPLEMENTATION REQUIRED HERE ---
#         # Implement the logic for the off-diagonal blocks.
#         # This is very similar to the non-causal version from Problem 3.
#         # 1. Load the K and V blocks for the current iteration.
#         k_offsets = start_n + tl.arange(0, BLOCK_N)
#         k_ptrs = K_ptr +  batch_idx * k_stride_b + head_idx * k_stride_h + \
#             (k_offsets[None, :] * k_stride_s + tl.arange(0, HEAD_DIM)[:, None])
#         k_block = tl.load(k_ptrs, mask = k_offsets[None, :] < SEQ_LEN , other = 0.0)

#         v_offsets = start_n + tl.arange(0, BLOCK_N)
#         v_ptrs = V_ptr + batch_idx * v_stride_b + head_idx * v_stride_h + \
#             (v_offsets[:, None] * v_stride_s + tl.arange(0, HEAD_DIM)[None, :])
#         v_block = tl.load(v_ptrs, mask = v_offsets[:, None] < SEQ_LEN, other = 0.0)

#         # 2. Compute the attention scores (S_ij).
#         S_ij = tl.dot(q_block, k_block)
#         ## mask = (start_n + k_offsets[:, None]) <= q_offsets[None, :] # don't quite get this part???
#         S_ij *= qk_scale ##+ tl.where(mask, 0, -1.0e6)
#         # 3. Update the online softmax statistics (m_i, l_i) and the accumulator (acc).
#         m_ij = tl.maximum(m_i, tl.max(S_ij, 1))
#         scale_factor = tl.exp2(m_i - m_ij)
#         P_ij = tl.exp2(S_ij - m_ij[:, None])
#         l_i = l_i * scale_factor + tl.sum(P_ij, 1)
#         acc = acc * scale_factor[:, None] + tl.dot(P_ij.to(v_block.type), v_block)

#         m_i = m_ij
#         pass
#         # --- END OF STUDENT IMPLEMENTATION ---


#     # --- Phase 2: Run on the Diagonal Blocks (With Masking) ---
#     # Process the blocks where query and key indices can overlap.
#     diag_start = q_block_idx * BLOCK_M
#     for start_n in range(diag_start, (q_block_idx + 1) * BLOCK_M, BLOCK_N):
#         # --- STUDENT IMPLEMENTATION REQUIRED HERE ---
#         # Implement the logic for the diagonal blocks, apply the causal mask to S_ij.
#         k_offsets = start_n + tl.arange(0, BLOCK_N)
#         k_ptrs = K_ptr + batch_idx * k_stride_b + head_idx * k_stride_h + \
#             (k_offsets[None, :] * k_stride_s + tl.arange(0, HEAD_DIM)[:, None])
#         k_block = tl.load(k_ptrs, mask = k_offsets[None, :] < SEQ_LEN , other = 0.0)

#         v_offsets = start_n + tl.arange(0, BLOCK_N)
#         v_ptrs = V_ptr + batch_idx * v_stride_b + head_idx * v_stride_h + \
#             (v_offsets[:, None] * v_stride_s + tl.arange(0, HEAD_DIM)[None, :])
#         v_block = tl.load(v_ptrs, mask = v_offsets[:, None] < SEQ_LEN, other = 0.0)

#         # 2. Compute the attention scores (S_ij).
#         S_ij = tl.dot(q_block, k_block)
#         mask = (k_offsets[None, :]) <= q_offsets[:, None] # don't quite get this part???
#         S_ij = tl.where(mask, S_ij, -float('inf'))
#         S_ij *= qk_scale
#         # 3. Update the online softmax statistics (m_i, l_i) and the accumulator (acc).
#         m_ij = tl.maximum(m_i, tl.max(S_ij, 1))
#         scale_factor = tl.exp2(m_i - m_ij) # * 1.44269504
#         P_ij = tl.exp2(S_ij - m_ij[:, None])
#         l_i = l_i * scale_factor + tl.sum(P_ij, 1)
#         acc = acc * scale_factor[:, None] + tl.dot(P_ij.to(v_block.type), v_block)

#         m_i = m_ij  
#         pass
#         # --- END OF STUDENT IMPLEMENTATION ---


#     # 4. Normalize and write the final output block.
#     l_i_safe = l_i[:, None] + 1e-6
#     acc = acc / l_i_safe
    
#     o_ptrs = O_ptr + batch_idx * q_stride_b + head_idx * q_stride_h + \
#              (q_offsets[:, None] * q_stride_s + tl.arange(0, HEAD_DIM)[None, :])
             
#     tl.store(o_ptrs, acc.to(O_ptr.dtype.element_ty), mask=q_offsets[:, None] < SEQ_LEN)


# @triton.jit
# def _flash_attention_forward_causal_kernel(
#     # Pointers to Tensors
#     Q_ptr, K_ptr, V_ptr, O_ptr,
#     # Stride information for tensors
#     q_stride_b, q_stride_h, q_stride_s,
#     k_stride_b, k_stride_h, k_stride_s,
#     v_stride_b, v_stride_h, v_stride_s,
#     # Kernel parameters
#     softmax_scale,
#     SEQ_LEN,
#     N_HEADS,
#     HEAD_DIM: tl.constexpr,
#     BLOCK_M: tl.constexpr,
#     BLOCK_N: tl.constexpr,
# ):
#     block_idx = tl.program_id(axis=0)
#     bh_idx = tl.program_id(axis=1)

#     # get query at bsize x nhead
#     b_idx = bh_idx // N_HEADS 
#     h_idx = bh_idx % N_HEADS 

#     q_offset = block_idx * BLOCK_M + tl.arange(0, BLOCK_M)
#     q_start = b_idx * q_stride_b + h_idx * q_stride_h + \
#         (q_offset[:, None] * q_stride_s + tl.arange(0, HEAD_DIM)[None,: ])

#     q_mask = q_offset[:, None] < SEQ_LEN
#     q_tile = tl.load(Q_ptr + q_start, mask=q_mask, other=0.0)

#     qk_scale = softmax_scale # ln-base

#     m_i = tl.full([BLOCK_M], -float('inf'), dtype = tl.float32)
#     l_i = tl.zeros([BLOCK_M], dtype = tl.float32)
#     acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype = tl.float32)

#     # loop over off-diagonal block of keys
#     for j in range(0, block_idx * BLOCK_M, BLOCK_N):
#         k_offsets = j + tl.arange(0, BLOCK_N)
#         k_start =  b_idx * k_stride_b + h_idx * k_stride_h + \
#             (tl.arange(0, HEAD_DIM)[:, None ] + k_offsets[None, :] * k_stride_s) # K.transpose(-1,-2)
#         k_mask = k_offsets[None, :] < SEQ_LEN 
#         k_tile = tl.load(K_ptr + k_start, mask=k_mask, other=0.0)
        
#         v_start =  b_idx * k_stride_b + h_idx * k_stride_h + \
#             (k_offsets[:, None] * k_stride_s + tl.arange(0, HEAD_DIM)[None,:])
#         v_mask = k_offsets[:, None] < SEQ_LEN 
#         v_tile = tl.load(V_ptr + v_start, mask=v_mask, other=0.0)
#     # calculate tiled attn_o
#         s_ij = tl.dot(q_tile, k_tile) * qk_scale
#         m_ij = tl.max(s_ij, axis = 1)
#         m_new = tl.maximum(m_i, m_ij)
#         scale_factor = tl.exp(m_i - m_new)
#         # correct
#         l_i *= scale_factor
#         acc *= scale_factor[:, None]
#         # update
#         p_tile = tl.exp(s_ij - m_new[:, None])
#         l_i += tl.sum(p_tile, axis = 1)
#         acc += tl.dot(p_tile.to(v_tile.type), v_tile)
#         m_i = m_new
#         # cn1 += 1


#     # loop over diagonal block of keys
#     diag_start = block_idx * BLOCK_M
#     for j in range(diag_start, (block_idx + 1) * BLOCK_M, BLOCK_N):
#         k_offsets = j + tl.arange(0, BLOCK_N)
#         k_start =  b_idx * k_stride_b + h_idx * k_stride_h + \
#             (tl.arange(0, HEAD_DIM)[:, None ] + k_offsets[None, :] * k_stride_s) # K.transpose(-1,-2)
#         k_mask = k_offsets[None, :] < SEQ_LEN 
#         k_tile = tl.load(K_ptr + k_start, mask=k_mask, other=0.0)
        
#         v_start =  b_idx * k_stride_b + h_idx * k_stride_h + \
#             (k_offsets[:, None] * k_stride_s + tl.arange(0, HEAD_DIM)[None,:])
#         v_mask = k_offsets[:, None] < SEQ_LEN 
#         v_tile = tl.load(V_ptr + v_start, mask=v_mask, other=0.0)
#     # calculate tiled attn_o
#         s_ij = tl.dot(q_tile, k_tile)
#         causal_mask = (q_offset[:, None] >= k_offsets[None,:])
#         s_ij = tl.where(causal_mask, s_ij, -float('inf'))
#         s_ij *= qk_scale
#         m_ij = tl.max(s_ij, axis = 1)
#         m_new = tl.maximum(m_i, m_ij)
#         scale_factor = tl.exp(m_i - m_new)
#         # correct
#         l_i *= scale_factor
#         acc *= scale_factor[:, None]
#         # update
#         p_tile = tl.exp(s_ij - m_new[:, None])
#         l_i += tl.sum(p_tile, axis = 1)
#         acc += tl.dot(p_tile.to(v_tile.type), v_tile)
#         m_i = m_new
#         # cnt2 += 1

#     # accumulate and return
#     l_i_safe = l_i[:, None] + 1e-6
#     acc = acc / l_i_safe

#     o_ptrs = b_idx * q_stride_b + h_idx * q_stride_h + \
#         (q_offset[:, None] * q_stride_s + tl.arange(0, HEAD_DIM)[None,: ])

#     tl.store(O_ptr + o_ptrs, acc, mask = q_mask)

# # Nov 18 practice
# @triton.jit
# def _flash_attention_forward_causal_kernel(
#     q_ptr, k_ptr, v_ptr, o_ptr,
#     q_stride_b, q_stride_h, q_stride_s,
#     k_stride_b, k_stride_h, k_stride_s,
#     v_stride_b, v_stride_h, v_stride_s,
#     softmax_scale,
#     SEQ_LEN, N_HEADS, HEAD_DIM: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N :tl.constexpr,
# ):
#     blk_idx = tl.program_id(axis = 0)
#     bh_idx = tl.program_id(axis = 1)

#     b_idx, h_idx = bh_idx // N_HEADS, bh_idx % N_HEADS
#     seq_idx = blk_idx * BLOCK_M

#     N_K_TILES = tl.cdiv(SEQ_LEN, BLOCK_N)

#     # get q_tile
#     q_off = blk_idx * BLOCK_M + tl.arange(0, BLOCK_M)
#     q_mask = q_off < SEQ_LEN

#     q_idx = b_idx * q_stride_b + h_idx * q_stride_h + q_off[:, None] * q_stride_s + tl.arange(0, HEAD_DIM)[None, :] #WRONG HERE: forgot to times q_off with q_stride_s !!!! basically forgot the sequence dimension
#     q_tile = tl.load(q_ptr + q_idx, mask = q_mask[:, None], other=0.0)

#     m_i = tl.full([BLOCK_M], -float('inf'), dtype = tl.float32) # WRONG HERE, forgot about -float('inf')
#     l_i = tl.zeros([BLOCK_M], dtype = tl.float32) # WRONG HERE: forgot syntax
#     o_i = tl.zeros([BLOCK_M, HEAD_DIM], dtype = tl.float32) # WRONG HERE: forgot syntax
    
#     # traverse across k, get k_tile, v_tile
#     # BLOCK1: Traverse to the first diagonal idx
#     FULL_BLOCK_TILE = tl.cdiv(seq_idx, BLOCK_N) 
#     for i in range(0, N_K_TILES): #FULL_BLOCK_TILE):
#         k_off = i * BLOCK_N + tl.arange(0, BLOCK_N)
#         k_mask = k_off < SEQ_LEN

#         k_idx = b_idx * k_stride_b + h_idx * k_stride_h + tl.arange(0, HEAD_DIM)[:, None] + k_off[None, :] * k_stride_s
#         k_tile = tl.load(k_ptr + k_idx, mask = k_mask[None, :], other=0.0) # WRONG HERE: k_mask unsqueeze wrong dimension

#         v_idx = b_idx * v_stride_b + h_idx * v_stride_h + k_off[:, None] * v_stride_s + tl.arange(0, HEAD_DIM)[None, :]
#         v_tile = tl.load(v_ptr + v_idx, mask = k_mask[:, None], other=0.0)

#         s_ij = tl.dot(q_tile, k_tile) * softmax_scale # inputs are float16, but output of tl.dot will be kept at fp32

#         causal_mask = q_off[:, None] >= k_off[None, :]
#         s_ij = tl.where(causal_mask, s_ij, -float('inf'))
    
#         # calculate running attn output
#         new_m = tl.max(s_ij, axis = -1)
#         new_m = tl.maximum(new_m, m_i)
#         scale_factor = tl.exp(m_i - new_m)
#         # correct 
#         l_i *= scale_factor
#         o_i *= scale_factor[:, None]
#         # update
#         p_tile = tl.exp(s_ij - new_m[:, None]) # e^-inf = 0
#         l_i += tl.sum(p_tile, axis = -1) # at positions of -inf, which is now 0 due to tl.exp operator. => won't add up to the progress
#         o_i += tl.dot(p_tile, v_tile.to(tl.float32))
#         # reassign
#         m_i = new_m

#     # # BLOCK2: Traverse along the diagonal block
#     # for i in range(FULL_BLOCK_TILE, N_K_TILES):
#     #     k_off = i * BLOCK_N + tl.arange(0, BLOCK_N)
#     #     k_mask = k_off < SEQ_LEN

#     #     k_idx = b_idx * k_stride_b + h_idx * k_stride_h + tl.arange(0, HEAD_DIM)[:, None] + k_off[None, :] * k_stride_s
#     #     k_tile = tl.load(k_ptr + k_idx, mask = k_mask[None, :], other=0.0) # WRONG HERE: k_mask unsqueeze wrong dimension

#     #     v_idx = b_idx * v_stride_b + h_idx * v_stride_h + k_off[:, None] * v_stride_s + tl.arange(0, HEAD_DIM)[None, :]
#     #     v_tile = tl.load(v_ptr + v_idx, mask = k_mask[:, None], other=0.0)

#     #     s_ij = tl.dot(q_tile, k_tile) * softmax_scale # inputs are float16, but output of tl.dot will be kept at fp32

#     #     causal_mask = q_off[:, None] >= k_off[None, :]
#     #     s_ij = tl.where(causal_mask, s_ij, -float('inf'))
    
#     #     # calculate running attn output
#     #     new_m = tl.max(s_ij, axis = -1)
#     #     new_m = tl.maximum(new_m, m_i)
#     #     scale_factor = tl.exp(m_i - new_m)
#     #     # correct 
#     #     l_i *= scale_factor
#     #     o_i *= scale_factor[:, None]
#     #     # update
#     #     p_tile = tl.exp(s_ij - new_m[:, None])
#     #     l_i += tl.sum(p_tile, axis = -1)
#     #     o_i += tl.dot(p_tile, v_tile.to(tl.float32))
#     #     # reassign
#     #     m_i = new_m
    
#     reciprocal_li = tl.where(l_i > 0, 1 / l_i, 0)
#     tl.store(o_ptr + q_idx, (o_i * reciprocal_li[:, None]).to(q_tile.dtype), mask= q_mask[:, None]) # WRONG HERE: forgot to broadcast reciprocal_li

# # Nov 20 practice
@triton.jit
def _flash_attention_forward_causal_kernel(
    q_ptr, k_ptr, v_ptr, o_ptr,
    q_stride_b, q_stride_h, q_stride_s,
    k_stride_b, k_stride_h, k_stride_s,
    v_stride_b, v_stride_h, v_stride_s,
    scale, SEQ_LEN, N_HEADS, HEAD_DIM : tl.constexpr, BLOCK_M : tl.constexpr, BLOCK_N : tl.constexpr
):
    blk_id = tl.program_id(axis = 0)
    bh_id = tl.program_id(axis = 1)

    b_idx, h_idx = bh_id // N_HEADS, bh_id % N_HEADS
    
    q_offs = blk_id * BLOCK_M + tl.arange(0, BLOCK_M) # haven't considered q_stride_s
    q_idx = b_idx * q_stride_b + h_idx * q_stride_h + q_offs[:, None] * q_stride_s + tl.arange(0, HEAD_DIM)[None, :] # [seq_len, h_dim]
    q_tile = tl.load(q_ptr + q_idx, mask = q_offs[:, None] < SEQ_LEN, other = 0.0)

    m_i = tl.full([BLOCK_M], -float('inf'), dtype = tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype = tl.float32)
    o_i = tl.zeros([BLOCK_M, HEAD_DIM], dtype = tl.float32)

    N_K_TILES = tl.cdiv(SEQ_LEN, BLOCK_N)
    processed_tiles = min(N_K_TILES, tl.cdiv((blk_id + 1) * BLOCK_M, BLOCK_N)) # PERF. NOTE: avoiding unnecessary blocks
    for i in range(processed_tiles):
        k_offs = i * BLOCK_N + tl.arange(0, BLOCK_N)
        k_idx = b_idx * k_stride_b + h_idx * k_stride_h + k_offs[None, : ] * k_stride_s + tl.arange(0, HEAD_DIM)[:, None] # [h_dim, seq_len]
        k_tile = tl.load(k_ptr + k_idx, mask = k_offs[None, :] < SEQ_LEN, other = 0.0)
        v_idx = b_idx * v_stride_b + h_idx * v_stride_h + k_offs[:, None ] * v_stride_s + tl.arange(0, HEAD_DIM)[None, :] # [seq_len, h_dim]
        v_tile = tl.load(v_ptr + v_idx, mask = k_offs[:, None] < SEQ_LEN, other = 0.0)

        s_ij = tl.dot(q_tile, k_tile) * scale # tl.dot returns float 32 in default
        
        # apply causal mask
        causal_mask = (q_offs[:, None] >= k_offs[None, :]) # WRONG Causal mask
        s_ij = tl.where(causal_mask, s_ij, -float('inf'))
        # correct 
        new_m = tl.max(s_ij, axis = -1)
        new_m = tl.maximum(new_m, m_i)
        scale_factor = tl.exp(m_i - new_m)
        l_i *= scale_factor
        o_i *= scale_factor[:, None] 
        # update
        p_tile = tl.exp(s_ij - new_m[:, None])
        l_i += tl.sum(p_tile, axis = -1)
        o_i += tl.dot(p_tile.to(v_tile.dtype), v_tile) # PERFORMANCE TRICK: cast p_tile to v_tile's, tl.dot() safely return float32 anyways  -- accumulator (fp32) = A (fp16) * B (fp16) + prev_acc (fp32)
        # Performance note:
        # no native fp32 tensor core (TC cannot multiply two full FP32 matrices directly). On Ampere/Hopper/Ada, Triton has to downcast both to TF32 (10 bits of mantissa ~ FP16's precision), but consumes 2x more register space and memory traffic
        # Tensor core performs best with the default FP16 types
        m_i = new_m

    normalized_li = tl.where(l_i > 0, 1 / l_i, 0.0)
    tl.store(o_ptr + q_idx, o_i * normalized_li[:, None], mask = q_offs[:, None] < SEQ_LEN) # tl.store doesn't have other 

def flash_attention_forward(q, k, v, is_causal=True):
    """
    Python wrapper for the single-kernel, two-phase causal FlashAttention.
    """
    if not is_causal:
        raise NotImplementedError("This implementation is for causal attention. Use solution_3 for non-causal.")

    batch, n_heads, seq_len, head_dim = q.shape
    o = torch.empty_like(q)
    softmax_scale = 1.0 / math.sqrt(head_dim)
    
    if head_dim >= 256:
        BLOCK_M, BLOCK_N = 64, 32
        num_stages = 1
    elif head_dim >= 128:
        BLOCK_M, BLOCK_N = 64, 64
        num_stages = 2
    else:
        BLOCK_M, BLOCK_N = 128, 64
        num_stages = 2

    grid = (triton.cdiv(seq_len, BLOCK_M), batch * n_heads)

    with nvtx.range("flash attn"):
        _flash_attention_forward_causal_kernel[grid](
            q, k, v, o,
            q.stride(0), q.stride(1), q.stride(2),
            k.stride(0), k.stride(1), k.stride(2),
            v.stride(0), v.stride(1), v.stride(2),
            softmax_scale,
            seq_len,
            n_heads,
            HEAD_DIM=head_dim,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            num_stages=num_stages,
        )
    return o

if __name__ == "__main__":
    Q = torch.randn(1, 10, 1024, 256, dtype=torch.float16, device="cuda")
    K = torch.randn(1, 10, 1024, 256, dtype=torch.float16, device="cuda")
    V = torch.randn(1, 10, 1024, 256, dtype=torch.float16, device="cuda")

    O = flash_attention_forward(Q, K, V)
    print("Flash attention executed successfully! Output shape:", O.shape)


# Performance 
# Casting o_i to q_tile.dtype is 0.8018 ms, while keeping it float32 is 1.2415
# because tl.dot returns float32 anyways, we can safely cast the operand p_tile to float16 to match v_tile for faster compute -> 0.7710 ms
# Peak memory all stay the same.