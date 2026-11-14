import torch
import triton
import triton.language as tl

@triton.jit
def weighted_row_sum_kernel(
    X_ptr,        # Pointer to the input tensor
    W_ptr,        # Pointer to the weight vector
    Y_ptr,        # Pointer to the output vector
    N_COLS,       # Number of columns in the input tensor
    BLOCK_SIZE: tl.constexpr  # Block size for the kernel
):
    """
    Triton kernel to compute the weighted sum of each row in a matrix.
    Y[i] = sum_{j=0}^{N_COLS-1} X[i, j] * W[j]
    """
    row_id = tl.program_id(axis=0)

    x_start = X_ptr + row_id * N_COLS

    accumulators = tl.zeros((BLOCK_SIZE, ), dtype = tl.float32 )
    for row_start_idx in range(0, tl.cdiv(N_COLS, BLOCK_SIZE)):
        offsets = row_start_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < N_COLS 
        X = tl.load(x_start + offsets, mask=mask, other = 0.0)
        W = tl.load(W_ptr + offsets, mask = mask, other = 0.0)

        accumulators += X * W 
    
    tl.store(Y_ptr + row_id, tl.sum(accumulators, axis = 0))
   
    
# --- END OF STUDENT IMPLEMENTATION ---


def weighted_row_sum_forward(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """
    Forward pass for the weighted row-sum operation using the Triton kernel.
    """
    assert x.dim() == 2, "Input tensor must be a 2D matrix"
    assert w.dim() == 1, "Weight tensor must be a 1D vector"
    assert x.shape[1] == w.shape[0], "Inner dimensions must match"
    assert x.is_cuda and w.is_cuda, "Tensors must be on CUDA"
    
    N_ROWS, N_COLS = x.shape
    
    # The output is a 1D tensor with length equal to the number of rows.
    y = torch.empty(N_ROWS, device=x.device, dtype=torch.float32)
    
    # The grid is 1D, with one program instance per row.
    grid = (N_ROWS,)
    
    # Block size is a power of 2. 1024 is a good default.
    BLOCK_SIZE = 1024
    
    # Launch the kernel
    weighted_row_sum_kernel[grid](
        x, w, y,
        N_COLS=N_COLS,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return y.to(x.dtype) # Cast back to original dtype

def torch_weighted_row_sum(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """
    Reference implementation using pure PyTorch.
    """
    return (x * w).sum(dim=1)