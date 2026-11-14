import torch
import triton
import triton.language as tl
import math

# @triton.jit
# def _flash_attention_forward_kernel(
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
#     """
#     Triton kernel for the forward pass of FlashAttention-2 (non-causal).
#     This is a template for student implementation.
#     """
#     # 1. Identify the block of queries and the batch/head to be processed.
#     q_block_idx = tl.program_id(axis=0)
#     batch_head_idx = tl.program_id(axis=1)
    
#     batch_idx = batch_head_idx // N_HEADS
#     head_idx = batch_head_idx % N_HEADS

#     # 2. Initialize pointers and accumulators for the online softmax.
#     m_i = tl.full([BLOCK_M], -float('inf'), dtype=tl.float32)
#     l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
#     acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

#     # 3. Load the block of queries (Q_i).
#     q_offsets = (q_block_idx * BLOCK_M + tl.arange(0, BLOCK_M))
#     q_ptrs = Q_ptr + batch_idx * q_stride_b + head_idx * q_stride_h + \
#              (q_offsets[:, None] * q_stride_s + tl.arange(0, HEAD_DIM)[None, :])
#     # a block of a [BATCH_NUM, HEAD_NUM, q_offsets[:, None] * q_stride_s :, :] => contains the unprocessed tokens, masking here is to only process the limit number of tokens
#     q_block = tl.load(q_ptrs, mask=q_offsets[:, None] < SEQ_LEN, other=0.0) 

#     # PyTorch softmax is exp(x), Triton is exp2(x * log2(e)), log2(e) is approx 1.44269504
#     qk_scale = softmax_scale # * 1.44269504

#     # 4. Main loop: Iterate over blocks of keys (K_j) and values (V_j).
#     for start_n in range(0, SEQ_LEN, BLOCK_N):
#         # - Load K_j
#         k_offsets = start_n + tl.arange(0, BLOCK_N)
#         k_ptrs = K_ptr + batch_idx * k_stride_b + head_idx * k_stride_h + \
#                  (k_offsets[None, :] * k_stride_s + tl.arange(0, HEAD_DIM)[:, None])
#         k_block = tl.load(k_ptrs, mask=k_offsets[None, :] < SEQ_LEN, other=0.0)
        
#         # Compute attention scores S_ij = Q_i * K_j^T
#         s_ij = tl.dot(q_block, k_block)
#         s_ij *= qk_scale

#         # Load V_j
#         v_ptrs = V_ptr + batch_idx * v_stride_b + head_idx * v_stride_h + \
#                  (k_offsets[:, None] * v_stride_s + tl.arange(0, HEAD_DIM)[None, :])
#         v_block = tl.load(v_ptrs, mask=k_offsets[:, None] < SEQ_LEN, other=0.0)

#         #  # --- STUDENT IMPLEMENTATION REQUIRED HERE ---
#         m_ij = tl.max(s_ij, axis = 1)
#         m_new = tl.maximum(m_i, m_ij)
#         # correct
#         scale_factor = tl.exp(m_i - m_new)
#         l_i = l_i * scale_factor
#         acc = acc * scale_factor[:, None]
#         # update
#         P_tilde_ij = tl.exp(s_ij - m_new[:, None])
#         l_i = l_i + tl.sum(P_tilde_ij, axis = 1)
#         acc = acc + tl.dot(P_tilde_ij.to(v_block.dtype), v_block)
#         m_i = m_new
#         # --- END OF STUDENT IMPLEMENTATION ---


#     # 5. Normalize the accumulator and write the output block.
#     # This part is provided. It handles the final normalization and write-back.
#     l_i_safe = l_i[:, None] + 1e-6
#     acc = acc / l_i_safe
    
#     o_ptrs = O_ptr + batch_idx * q_stride_b + head_idx * q_stride_h + \
#              (q_offsets[:, None] * q_stride_s + tl.arange(0, HEAD_DIM)[None, :])
             
#     tl.store(o_ptrs, acc.to(O_ptr.dtype.element_ty), mask=q_offsets[:, None] < SEQ_LEN)

@triton.jit
def _flash_attention_forward_kernel(
    # Pointers to Tensors
    Q_ptr, K_ptr, V_ptr, O_ptr,
    # Stride information for tensors
    q_stride_b, q_stride_h, q_stride_s,
    k_stride_b, k_stride_h, k_stride_s,
    v_stride_b, v_stride_h, v_stride_s,
    # Kernel parameters
    softmax_scale,
    SEQ_LEN,
    N_HEADS,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    block_idx = tl.program_id(axis=0)
    bh_idx = tl.program_id(axis=1)

    # get query at bsize x nhead
    b_idx = bh_idx // N_HEADS 
    h_idx = bh_idx % N_HEADS 

    q_offset = block_idx * BLOCK_M + tl.arange(0, BLOCK_M)
    q_start = b_idx * q_stride_b + h_idx * q_stride_h + \
        (q_offset[:, None] * q_stride_s + tl.arange(0, HEAD_DIM)[None,: ])

    q_mask = q_offset[:, None] < SEQ_LEN
    q_tile = tl.load(Q_ptr + q_start, mask=q_mask, other=0.0)

    qk_scale = softmax_scale # ln-base

    m_i = tl.full([BLOCK_M], -float('inf'), dtype = tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype = tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype = tl.float32)

    # loop over block of keys
    for j in range(0, tl.cdiv(SEQ_LEN, BLOCK_N)): # traverse the sequence dimension of key-val
        k_offsets = j * BLOCK_N + tl.arange(0, BLOCK_N)
        k_start =  b_idx * k_stride_b + h_idx * k_stride_h + \
            (tl.arange(0, HEAD_DIM)[:, None ] + k_offsets[None, :] * k_stride_s) # K.transpose(-1,-2)
        k_mask = k_offsets[None, :] < SEQ_LEN 
        k_tile = tl.load(K_ptr + k_start, mask=k_mask, other=0.0)
        
        v_start =  b_idx * k_stride_b + h_idx * k_stride_h + \
            (k_offsets[:, None] * k_stride_s + tl.arange(0, HEAD_DIM)[None,:])
        v_mask = k_offsets[:, None] < SEQ_LEN 
        v_tile = tl.load(V_ptr + v_start, mask=v_mask, other=0.0)
    # calculate tiled attn_o
        s_ij = tl.dot(q_tile, k_tile) * qk_scale
        m_ij = tl.max(s_ij, axis = 1)
        m_new = tl.maximum(m_i, m_ij)
        scale_factor = tl.exp(m_i - m_new)
        # correct
        l_i *= scale_factor
        acc *= scale_factor[:, None]
        # update
        p_tile = tl.exp(s_ij - m_new[:, None])
        l_i += tl.sum(p_tile, axis = 1)
        acc += tl.dot(p_tile.to(v_tile.type), v_tile)
        m_i = m_new
    # accumulate and return
    l_i_safe = l_i[:, None] + 1e-6
    acc = acc / l_i_safe

    o_ptrs = b_idx * q_stride_b + h_idx * q_stride_h + \
        (q_offset[:, None] * q_stride_s + tl.arange(0, HEAD_DIM)[None,: ])

    tl.store(O_ptr + o_ptrs, acc, mask = q_mask)


def flash_attention_forward(q, k, v, is_causal=False):
    """
    Minimal Python wrapper for the FlashAttention-2 forward pass.
    """
    batch, n_heads, seq_len, head_dim = q.shape
    o = torch.empty_like(q)
    softmax_scale = 1.0 / math.sqrt(head_dim)
    BLOCK_M, BLOCK_N = 128, 64
    grid = (triton.cdiv(seq_len, BLOCK_M), batch * n_heads)

    _flash_attention_forward_kernel[grid](
        q, k, v, o,
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        softmax_scale,
        seq_len,
        n_heads,
        HEAD_DIM=head_dim,
        BLOCK_M=BLOCK_M, #size of each Q block 
        BLOCK_N=BLOCK_N, #size of each KV block
    )
    return o