/******************************************************************************
 * Copyright (c) 2023, lsl.
 ******************************************************************************/

#include "warp_traits.h"
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>

#define CHECK_DEVICE(x) TORCH_CHECK(x.device().type() == torch::kCUDA, #x " must be on CUDA")
#define CHECK_SHAPE(x, ...) TORCH_CHECK(x.sizes() == torch::IntArrayRef({__VA_ARGS__}), #x " must have shape (" #__VA_ARGS__ ")")

void matrix_v8_cuda(const torch::Tensor C, const torch::Tensor A, torch::Tensor B);

// x1: RowMajor, x2: RowMajor, out: RowMajor
void matrix_v8(const torch::Tensor C, const torch::Tensor A, torch::Tensor B) {
    CHECK_DEVICE(C);
    CHECK_DEVICE(A);
    CHECK_DEVICE(B);

    // auto elem_type = half;
    TORCH_CHECK(B.dtype() == B.dtype());

    // TORCH_CHECK(elem_type == half);

    int M = A.sizes()[0];
    int N = B.sizes()[1];
    int K = A.sizes()[1];

    TORCH_CHECK(B.sizes()[0] == K);
    TORCH_CHECK(C.sizes()[0] == M);
    TORCH_CHECK(C.sizes()[1] == N);

    // Otherwise the kernel will be launched from cuda:0 device
    // Cast to char to avoid compiler warning about narrowing
    at::cuda::CUDAGuard device_guard{(char)A.get_device()};
    
    constexpr static int ATOM_M = 16;
    constexpr static int ATOM_N = 8;
    constexpr static int ATOM_K = 16;
    constexpr static int kTile_M = 16;
    constexpr static int kTile_N = 16;
    constexpr static int kTile_K = 16;
    constexpr static int Warp_M = 1;
    constexpr static int Warp_N = 1;
    constexpr static int Warp_K = 1;

    TORCH_CHECK(M % kTile_M == 0);
    TORCH_CHECK(N % kTile_N == 0);
    TORCH_CHECK(K % kTile_K == 0);

    matrix_v8_cuda<Warp_traits<ATOM_M, ATOM_N, ATOM_K, kTile_M, kTile_N, kTile_K, Warp_M, Warp_N, Warp_K, __half>>(C, A, B, M, N, K);
    // using tmp = Warp_traits<16>;
    matrix_v8_cuda<tmp>(C, A, B, M, N, K);
}


