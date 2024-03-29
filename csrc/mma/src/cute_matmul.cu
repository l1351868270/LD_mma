
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>

#include "cute/arch/mma_sm80.hpp"

#include "cute_matmul_v1.h"
#include "cute_traits_v1.h"
#include "cute_matmul_v2.h"
#include "cute_traits_v2.h"
#include "cute_matmul_v3.h"
#include "cute_traits_v3.h"
#include "cute_matmul_v4.h"
#include "cute_traits_v4.h"

// C: row-major 
// A: row-major 
// B: col-major
void cute_matmul_v1(const torch::Tensor C, const torch::Tensor A, torch::Tensor B) {
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
    

    constexpr static int kTile_M = 16;
    constexpr static int kTile_N = 16;
    constexpr static int kTile_K = 16;
    constexpr static int Warp_M = 1;
    constexpr static int Warp_N = 1;
    constexpr static int Warp_K = 1;

    TORCH_CHECK(M % kTile_M == 0);
    TORCH_CHECK(N % kTile_N == 0);
    TORCH_CHECK(K % kTile_K == 0);

    cute_matmul_v1_cuda<Cute_traits<kTile_M, kTile_N, kTile_K, Warp_M, Warp_N, Warp_K>>(C, A, B, M, N, K);
}

// C: row-major 
// A: row-major 
// B: col-major
void cute_matmul_v2(const torch::Tensor C, const torch::Tensor A, torch::Tensor B) {
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
    

    constexpr static int kTile_M = 16;
    constexpr static int kTile_N = 16;
    constexpr static int kTile_K = 16;
    constexpr static int Warp_M = 1;
    constexpr static int Warp_N = 1;
    constexpr static int Warp_K = 1;

    TORCH_CHECK(M % kTile_M == 0);
    TORCH_CHECK(N % kTile_N == 0);
    TORCH_CHECK(K % kTile_K == 0);

    cute_matmul_v2_cuda<Cute_traits_v2<kTile_M, kTile_N, kTile_K, Warp_M, Warp_N, Warp_K>>(C, A, B, M, N, K);
}

// C: row-major 
// A: row-major 
// B: col-major
void cute_matmul_v3(const torch::Tensor C, const torch::Tensor A, torch::Tensor B) {
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

    cute_matmul_v3_cuda<Cute_traits_v3<kTileM, kTileN, kTileK, WarpM, WarpN, WarpK>>(C, A, B, M, N, K);
}

void cute_matmul_v4(const torch::Tensor C, const torch::Tensor A, torch::Tensor B) {
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
    

    constexpr static int kTile_M = 16 * 4;
    constexpr static int kTile_N = 16 * 4;
    constexpr static int kTile_K = 16 * 16;
    constexpr static int Warp_M = 1 * 4;
    constexpr static int Warp_N = 1;
    constexpr static int Warp_K = 1;

    TORCH_CHECK(M % kTile_M == 0);
    TORCH_CHECK(N % kTile_N == 0);
    TORCH_CHECK(K % kTile_K == 0);

    cute_matmul_v4_cuda<Cute_traits_v4<kTile_M, kTile_N, kTile_K, Warp_M, Warp_N, Warp_K>>(C, A, B, M, N, K);
}