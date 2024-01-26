/******************************************************************************
 * Copyright (c) 2023, lsl.
 ******************************************************************************/

#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>

void warp_matmul_v8(const torch::Tensor C, const torch::Tensor A, torch::Tensor B);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("cpp_warp_matmul_v8", &warp_matmul_v8, "lsl warp matmul v8");
}