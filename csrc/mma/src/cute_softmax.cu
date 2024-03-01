#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>

#include "cute_softmax_v1.h"
#include "cute_softmax_traits_v1.h"

void cute_softmax_v1(const torch::Tensor C) {

    // TORCH_CHECK(elem_type == half);

    int M = C.sizes()[0];
    int N = C.sizes()[1];

    // Otherwise the kernel will be launched from cuda:0 device
    // Cast to char to avoid compiler warning about narrowing
    at::cuda::CUDAGuard device_guard{(char)C.get_device()};
    
    constexpr static int kTileM = 16 * 4;
    constexpr static int kTileN = 16 * 4;
    constexpr static int WarpM = 1 * 4;
    constexpr static int WarpN = 1;

    TORCH_CHECK(M % kTileM == 0);
    TORCH_CHECK(N % kTileN == 0);

    cute_softmax_v1_cuda<Cute_softmax_traits_v1<kTileM, kTileN, WarpM, WarpN>>(C, M, N);
}