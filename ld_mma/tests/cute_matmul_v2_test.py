import torch
import ld_mma
from ld_mma.ld_mma_interface import cute_matmul_v2

torch.set_printoptions (precision=6)
torch.manual_seed(0)

M = 16 * 4
N = 16 * 3
K = 16 * 1024# can not large than 16384

device = 'cuda'
dtype = torch.float16

A = torch.randn((M, K), device='cuda', dtype=dtype, requires_grad=False)
# A_range = torch.range(0, M*K-1, device='cuda', dtype=dtype, requires_grad=False)

# print(A_range)
# A = torch.reshape(torch.range(0, M*K-1, device='cuda', dtype=dtype, requires_grad=False), (M, K))
B = torch.randn((N, K), device='cuda', dtype=dtype, requires_grad=False)
# B = B.transpose(0, 1).contiguous()
C = torch.empty((M, N), device='cuda', dtype=dtype, requires_grad=False)

cute_matmul_v2(C, A, B)
print(f'C: {C}')

BT = B.transpose(0, 1).contiguous()
D = torch.matmul(A, BT)
print(f'D: {D}')