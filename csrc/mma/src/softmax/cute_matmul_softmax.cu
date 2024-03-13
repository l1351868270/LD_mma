

#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>

#include "cute/arch/mma_sm80.hpp"

#include "softmax/cute_matmul_softmax_v1.h"
#include "softmax/cute_matmul_softmax_traits_v1.h"

// C: row-major 
// A: row-major 
// B: col-major
void cute_matmul_softmax_v1(const torch::Tensor C, const torch::Tensor A, torch::Tensor B) {
    // auto elem_type = half;

    TORCH_CHECK(A.dtype() == B.dtype());

    // TORCH_CHECK(elem_type == half);

    int M = A.sizes()[0];
    int N = B.sizes()[0];
    int K = A.sizes()[1];

    TORCH_CHECK(B.sizes()[1] == K);
    TORCH_CHECK(C.sizes()[0] == M);
    TORCH_CHECK(C.sizes()[1] == N);

    // Otherwise the kernel will be launched from cuda:0 device
    // Cast to char to avoid compiler warning about narrowing
    at::cuda::CUDAGuard device_guard{(char)A.get_device()};
    

    constexpr static int kTileM = 16 * 4;
    constexpr static int kTileN = 16 * 4;
    constexpr static int kTileK = 16 * 4;
    constexpr static int WarpM = 1 * 4;
    constexpr static int WarpN = 1;
    constexpr static int WarpK = 1;

    TORCH_CHECK(M % kTileM == 0);
    TORCH_CHECK(N % kTileN == 0);
    TORCH_CHECK(K % kTileK == 0);

    cute_matmul_softmax_v1_cuda<Cute_matmul_softmax_traits_v1<kTileM, kTileN, kTileK, WarpM, WarpN, WarpK>>(C, A, B, M, N, K);
}