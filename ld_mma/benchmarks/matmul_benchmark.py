import time
import torch
import ld_mma
from ld_mma.ld_mma_interface import cublas_matmul, warp_matmul, cute_matmul_v1
from ld_mma.ld_mma_interface import cute_matmul_v2, cute_matmul_v3, cute_matmul_v4
torch.set_printoptions (precision=6)
torch.manual_seed(0)

# M = 81920 # 64 * 1280   64 * 1280
# N = 256   # 64 * 4      256 * 1
# K = 256   # 64 * 4      64 * 4

M = 2560
N = 2560
K = 64 * 4

device = 'cuda'
dtype = torch.float16

A = torch.randn((M, K), device='cuda', dtype=dtype, requires_grad=False)
B = torch.randn((N, K), device='cuda', dtype=dtype, requires_grad=False)
BT = torch.transpose_copy(B, 0, 1)

cublas_C = torch.empty((M, N), device='cuda', dtype=dtype, requires_grad=False)
# warp_C = torch.empty((M, N), device='cuda', dtype=dtype, requires_grad=False)
# cute_v1_C = torch.empty((M, N), device='cuda', dtype=dtype, requires_grad=False)
# cute_v2_C = torch.empty((M, N), device='cuda', dtype=dtype, requires_grad=False)
cute_v3_C = torch.empty((M, N), device='cuda', dtype=dtype, requires_grad=False)
cute_v4_C = torch.empty((M, N), device='cuda', dtype=dtype, requires_grad=False)

nt = 10
start = time.time()
for i in range(nt):
    cublas_matmul(cublas_C, A, B)
    print(f'cublas_matmul C: {cublas_C}')
    # warp_matmul(warp_C, A, BT)
    # print(f'warp_matmul C: {warp_C}')
    # cute_matmul_v1(cute_v1_C, A, B)
    # print(f'cute_matmul_v1 C: {cute_v1_C}')
    # cute_matmul_v2(cute_v2_C, A, B)
    # print(f'cute_matmul_v2 C: {cute_v2_C}')
    cute_matmul_v3(cute_v3_C, A, B)
    print(f'cute_matmul_v3 C: {cute_v3_C}')
    cute_matmul_v4(cute_v4_C, A, B)
    print(f'cute_matmul_v4 C: {cute_v4_C}')

tt = (time.time() - start) * 1000
time.sleep(3)
print(f"cublas_matmul: M={M}, N={N}, K={K}, nt={nt},  total time: {tt:.0f}ms, average time: {tt/nt:.0f}ms")

