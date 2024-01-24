/******************************************************************************
 * Copyright (c) 2023, lsl.
 ******************************************************************************/

#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>

#define CHECK_DEVICE(x) TORCH_CHECK(x.device().type() == torch::kCUDA, #x " must be on CUDA")
#define CHECK_SHAPE(x, ...) TORCH_CHECK(x.sizes() == torch::IntArrayRef({__VA_ARGS__}), #x " must have shape (" #__VA_ARGS__ ")")

void matrix_v8_cuda(const torch::Tensor x1, const torch::Tensor x2, torch::Tensor out);

// x1: RowMajor, x2: RowMajor, out: RowMajor
void matrix_v8(const torch::Tensor x1, const torch::Tensor x2, torch::Tensor out) {
    CHECK_DEVICE(x1); CHECK_DEVICE(x2);
    CHECK_DEVICE(x2); CHECK_DEVICE(x2);
    CHECK_DEVICE(out); CHECK_DEVICE(out);
    TORCH_CHECK(x1.dtype() == x2.dtype());
    // TORCH_CHECK(x1.dtype() == out.dtype());

    TORCH_CHECK(x1.sizes()[1] == x2.sizes()[0]);
    TORCH_CHECK(x1.sizes()[0] == out.sizes()[0]);
    TORCH_CHECK(x2.sizes()[1] == out.sizes()[1]);
    // torch::T
    

    // Otherwise the kernel will be launched from cuda:0 device
    // Cast to char to avoid compiler warning about narrowing
    at::cuda::CUDAGuard device_guard{(char)x1.get_device()};

    matrix_v8_cuda(x1, x2, out);
}


