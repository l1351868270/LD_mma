# python ./ld_mma/tests/cute_softmax_v1_test.py
# nsys profile python ./ld_mma/tests/cute_softmax_v1_test.py
# ncu -f --set full -o cute_matmul_v3 python ./ld_mma/tests/cute_softmax_v1_test.py

import torch
import ld_mma
from ld_mma.ld_mma_interface import cute_softmax_v1

torch.set_printoptions (precision=6)
torch.manual_seed(0)

M = 16 * 4 * 3
N = 16 * 4 * 5

device = 'cuda'
dtype = torch.float16

C = torch.empty((M, N), device='cuda', dtype=dtype, requires_grad=False)
# for i in range(N):
#     print(B[i])
cute_softmax_v1(C)

# print(f'C: {C}')