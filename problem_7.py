
import torch
import triton
import triton.language as tl
import math


@triton.jit
def _flash_attention_forward_swa_kernel(
    # Pointers to Tensors
    Q_ptr, K_ptr, V_ptr, O_ptr,
    # Stride information for tensors
    q_stride_b, q_stride_h, q_stride_s,
    k_stride_b, k_stride_h, k_stride_s,
    v_stride_b, v_stride_h, v_stride_s,
    # Kernel parameters
    softmax_scale,
    SEQ_LEN,
    N_Q_HEADS,
    N_KV_HEADS,
    WINDOW_SIZE: tl.constexpr,
    SINK_SIZE: tl.constexpr,
    # Constexpr tile sizes
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """
    Triton kernel for the forward pass of causal FlashAttention
    with GQA, Sliding Window Attention, and Attention Sink.
    """
    # 1. Identify the block of queries and the batch/head to be processed.
    q_block_idx = tl.program_id(axis=0)
    batch_head_idx = tl.program_id(axis=1)

    batch_idx = batch_head_idx // N_Q_HEADS
    q_head_idx = batch_head_idx % N_Q_HEADS

    # --- GQA Logic: Map Query Head to Shared K/V Head ---
    num_groups = N_Q_HEADS // N_KV_HEADS
    kv_head_idx = q_head_idx // num_groups

    # 2. Initialize accumulators in SRAM.
    m_i = tl.full([BLOCK_M], -float('inf'), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    # 3. Load the block of queries (Q_i).
    q_offsets = q_block_idx * BLOCK_M + tl.arange(0, BLOCK_M)
    q_ptrs = Q_ptr + batch_idx * q_stride_b + q_head_idx * q_stride_h + \
             (q_offsets[:, None] * q_stride_s + tl.arange(0, HEAD_DIM)[None, :])
    q_block = tl.load(q_ptrs, mask=q_offsets[:, None] < SEQ_LEN, other=0.0)

    qk_scale = softmax_scale * 1.44269504

    # Calculate the sliding window boundaries
    diag_start = q_block_idx * BLOCK_M
    diag_end = tl.minimum(diag_start + BLOCK_M, SEQ_LEN)

    window_start = tl.maximum(0, diag_start - WINDOW_SIZE + 1)
    first_window_key = tl.maximum(SINK_SIZE, window_start)
    first_window_block = (first_window_key // BLOCK_N) * BLOCK_N

    # --- Phase 0: Sink Attention ---
    for start_n in range(0, SINK_SIZE, BLOCK_N):
        k_offsets = start_n + tl.arange(0, BLOCK_N)
        k_ptrs = K_ptr + batch_idx * k_stride_b + kv_head_idx * k_stride_h + \
                 (k_offsets[None, :] * k_stride_s + tl.arange(0, HEAD_DIM)[:, None])
        k_block = tl.load(k_ptrs, mask=k_offsets[None, :] < SEQ_LEN, other=0.0)

        v_offsets = start_n + tl.arange(0, BLOCK_N)
        v_ptrs = V_ptr + batch_idx * v_stride_b + kv_head_idx * v_stride_h + \
                 (v_offsets[:, None] * v_stride_s + tl.arange(0, HEAD_DIM)[None, :])
        v_block = tl.load(v_ptrs, mask=v_offsets[:, None] < SEQ_LEN, other=0.0)

        # Apply sink mask (sink tokens must also be causal)
        S_ij = tl.dot(q_block, k_block)

        sink_mask = (
            (k_offsets[None, :] < SINK_SIZE)
            & (k_offsets[None, :] < SEQ_LEN)
            & (k_offsets[None, :] <= q_offsets[:, None])
        )

        S_ij *= qk_scale
        S_ij = tl.where(sink_mask, S_ij, -float('inf'))

        # Online softmax update
        m_ij = tl.maximum(m_i, tl.max(S_ij, 1))
        m_ij_safe = tl.where(m_ij == -float('inf'), 0.0, m_ij)
        scale_factor = tl.where(
            m_i == -float('inf'), 0.0, tl.exp2(m_i - m_ij_safe)
        )
        P_ij = tl.exp2(S_ij - m_ij_safe[:, None])

        l_i = l_i * scale_factor + tl.sum(P_ij, 1)
        acc = acc * scale_factor[:, None] + tl.dot(P_ij.to(v_block.type), v_block)
        m_i = m_ij

    # --- Phase 1: Sliding Window Attention (including diagonal blocks) ---
    for start_n in range(first_window_block, diag_end, BLOCK_N):
        k_offsets = start_n + tl.arange(0, BLOCK_N)
        k_ptrs = K_ptr + batch_idx * k_stride_b + kv_head_idx * k_stride_h + \
                 (k_offsets[None, :] * k_stride_s + tl.arange(0, HEAD_DIM)[:, None])
        k_block = tl.load(k_ptrs, mask=k_offsets[None, :] < SEQ_LEN, other=0.0)

        v_offsets = start_n + tl.arange(0, BLOCK_N)
        v_ptrs = V_ptr + batch_idx * v_stride_b + kv_head_idx * v_stride_h + \
                 (v_offsets[:, None] * v_stride_s + tl.arange(0, HEAD_DIM)[None, :])
        v_block = tl.load(v_ptrs, mask=v_offsets[:, None] < SEQ_LEN, other=0.0)

        # Apply sliding window mask
        # delta = i - j; valid iff 0 <= delta < WINDOW_SIZE
        S_ij = tl.dot(q_block, k_block)

        delta = q_offsets[:, None] - k_offsets[None, :]
        sw_mask = (
            (k_offsets[None, :] >= SINK_SIZE)
            & (k_offsets[None, :] < SEQ_LEN)
            & (delta >= 0)
            & (delta < WINDOW_SIZE)
        )

        S_ij *= qk_scale
        S_ij = tl.where(sw_mask, S_ij, -float('inf'))

        # Online softmax update
        m_ij = tl.maximum(m_i, tl.max(S_ij, 1))
        m_ij_safe = tl.where(m_ij == -float('inf'), 0.0, m_ij)
        scale_factor = tl.where(
            m_i == -float('inf'), 0.0, tl.exp2(m_i - m_ij_safe)
        )
        P_ij = tl.exp2(S_ij - m_ij_safe[:, None])

        l_i = l_i * scale_factor + tl.sum(P_ij, 1)
        acc = acc * scale_factor[:, None] + tl.dot(P_ij.to(v_block.type), v_block)
        m_i = m_ij

    # 4. Normalize and write the final output block.
    l_i_safe = tl.where(l_i == 0, 1.0, l_i)
    acc = acc / l_i_safe[:, None]

    o_ptrs = O_ptr + batch_idx * q_stride_b + q_head_idx * q_stride_h + \
             (q_offsets[:, None] * q_stride_s + tl.arange(0, HEAD_DIM)[None, :])

    tl.store(o_ptrs, acc.to(O_ptr.dtype.element_ty), mask=q_offsets[:, None] < SEQ_LEN)


def flash_attention_forward(q, k, v, is_causal=True, window_size=128, sink_size=4):
    """
    Python wrapper for the SWA-enabled GQA causal FlashAttention kernel with attention sink support.
    """
    # Shape checks
    batch, n_q_heads, seq_len, head_dim = q.shape
    _, n_kv_heads, _, _ = k.shape

    # Assertions
    assert q.shape[0] == v.shape[0] and q.shape[2] == v.shape[2] and q.shape[3] == v.shape[3]
    assert k.shape == v.shape
    assert head_dim <= 128
    assert n_q_heads % n_kv_heads == 0
    assert is_causal, "This kernel only supports causal attention"
    assert window_size > 0
    assert sink_size >= 0

    o = torch.empty_like(q)
    softmax_scale = 1.0 / math.sqrt(head_dim)

    BLOCK_M, BLOCK_N = 128, 64
    grid = (triton.cdiv(seq_len, BLOCK_M), batch * n_q_heads)

    _flash_attention_forward_swa_kernel[grid](
        q, k, v, o,
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        softmax_scale,
        seq_len,
        n_q_heads,
        n_kv_heads,
        WINDOW_SIZE=window_size,
        SINK_SIZE=sink_size,
        HEAD_DIM=head_dim,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
    )
    return o
