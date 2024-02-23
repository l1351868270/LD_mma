
import torch
import ld_mma_cuda

def bank_conflicts_v1(offset: int):
    ld_mma_cuda.cpp_bank_conflicts_v1(offset)

def bank_conflicts_v2(offset: int):
    ld_mma_cuda.cpp_bank_conflicts_v2(offset)

def bank_conflicts_v3(offset: int):
    ld_mma_cuda.cpp_bank_conflicts_v3(offset)

def bank_conflicts_v4(offset: int):
    ld_mma_cuda.cpp_bank_conflicts_v4(offset)

def bank_conflicts_v5(offset: int):
    ld_mma_cuda.cpp_bank_conflicts_v5(offset)

def bank_conflicts_v6(offset: int):
    ld_mma_cuda.cpp_bank_conflicts_v6(offset)

def cublas_matmul(C: torch.Tensor, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    # M = A.shape[0]
    # N = B.shape[1]
    # K = A.shape[1]
    # if B.shape[0] != K:
    #     raise Exception(f'A.shape[1] must equal B.shape[0]')
    # C = torch.empty((M, N), )
    ld_mma_cuda.cpp_cublas_matmul(C, A, B)


def warp_matmul(C: torch.Tensor, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    # M = A.shape[0]
    # N = B.shape[1]
    # K = A.shape[1]
    # if B.shape[0] != K:
    #     raise Exception(f'A.shape[1] must equal B.shape[0]')
    # C = torch.empty((M, N), )
    ld_mma_cuda.cpp_warp_matmul_v8(C, A, B)


# C: row-major 
# A: row-major 
# B: col-major
def cute_matmul_v1(C: torch.Tensor, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    ld_mma_cuda.cpp_cute_matmul_v1(C, A, B)

def cute_matmul_v2(C: torch.Tensor, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    ld_mma_cuda.cpp_cute_matmul_v2(C, A, B)

def cute_matmul_v3(C: torch.Tensor, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    ld_mma_cuda.cpp_cute_matmul_v3(C, A, B)