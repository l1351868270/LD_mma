import time
import torch
import ld_mma
from ld_mma.ld_mma_interface import cublas_matmul, cute_matmul_v2

torch.set_printoptions (precision=6)
torch.manual_seed(0)

M = 81920
N = 256
K = 256# can not large than 16384

device = 'cuda'
dtype = torch.float16

A = torch.randn((M, K), device='cuda', dtype=dtype, requires_grad=False)

B = torch.randn((N, K), device='cuda', dtype=dtype, requires_grad=False)

C = torch.empty((M, N), device='cuda', dtype=dtype, requires_grad=False)

nt = 1
start = time.time()
for i in range(nt):
    cute_matmul_v2(C, A, B)
    print(f'C: {C}')
tt = (time.time() - start) * 1000
print(f"cublas_matmul: M={M}, N={N}, K={K}, nt={nt},  total time: {tt:.0f}ms, average time: {tt/nt:.0f}ms")

# BT = B.transpose(0, 1).contiguous()
# D = torch.matmul(A, BT)
# print(f'D: {D}')