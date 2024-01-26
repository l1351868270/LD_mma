
import torch
import ld_mma
from ld_mma.ld_mma_interface import warp_matmul

M = 16
N = 8
K = 16

device = 'cuda'
dtype = torch.float16

A = torch.randn((M, K), device='cuda', dtype=dtype, requires_grad=False)
B = torch.randn((K, N), device='cuda', dtype=dtype, requires_grad=False)
C = torch.empty((M, N), device='cuda', dtype=dtype, requires_grad=False)

warp_matmul(C, A, B)

print(f'C: {C}')