
import torch
import ld_mma_cuda


def warp_matmul(C: torch.Tensor, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    # M = A.shape[0]
    # N = B.shape[1]
    # K = A.shape[1]
    # if B.shape[0] != K:
    #     raise Exception(f'A.shape[1] must equal B.shape[0]')
    # C = torch.empty((M, N), )
    ld_mma_cuda.cpp_warp_matmul_v8(C, A, B)