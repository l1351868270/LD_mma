
import torch
import ld_mma
from ld_mma.ld_mma_interface import cute_matmul_v1

torch.set_printoptions (precision=6)
torch.manual_seed(0)

M = 16 * 2
N = 16 * 3
K = 16 * 1024 # can not large than 16384

device = 'cuda'
dtype = torch.float16

A = torch.randn((M, K), device='cuda', dtype=dtype, requires_grad=False)
B = torch.randn((N, K), device='cuda', dtype=dtype, requires_grad=False)
# B = B.transpose(0, 1).contiguous()
C = torch.empty((M, N), device='cuda', dtype=torch.float32, requires_grad=False)

cute_matmul_v1(C, A, B)

print(f'C: {C}')

B = B.transpose(0, 1).contiguous()
D = torch.matmul(A, B)

print(f'D: {D}')