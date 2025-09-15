import torch
import triton 
import triton.language as tl

@triton.jit
def row_sum_kernel(
    X_ptr, Output_ptr,
    M, D,
    BLOCK_SIZE_D: tl.constexpr
):
    # get the unique row index for this program instance
    pid_m = tl.program_id(axis = 0)
    # calculate the starting pointer for the assigned row in X
    row_start_ptr = X_ptr + pid_m * D
    # initialize an accumulator in SRAM for the sum
    accumulator = 0.0

    print("pid_m ", pid_m, "; its dtype: ", pid_m.dtpye)
    print("row_start_ptr ", row_start_ptr, "; its dtype: ", row_start_ptr.dtype)

    #loop over the row in blocks (tiles).
    for d_offset in range(0, tl.cdiv(D, BLOCK_SIZE_D)):
        # create offsets and a mask for the current tile.
        d_offsets = d_offset * BLOCK_SIZE_D  + tl.arange(0, BLOCK_SIZE_D)
        d_mask = d_offsets < D
    return 0