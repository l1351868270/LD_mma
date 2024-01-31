
import torch
import ld_mma
from ld_mma.ld_mma_interface import cute_matmul

torch.set_printoptions (precision=6)
torch.manual_seed(0)

M = 16 * 1
N = 8 * 2
K = 16 * 1

device = 'cuda'
dtype = torch.float16

A = torch.ones((M, K), device='cuda', dtype=dtype, requires_grad=False)
B = torch.ones((K, N), device='cuda', dtype=dtype, requires_grad=False)
C = torch.empty((M, N), device='cuda', dtype=dtype, requires_grad=False)

cute_matmul(C, A, B)

print(f'C: {C}')

D = torch.matmul(A, B)

print(f'D: {D}')