/******************************************************************************
 * Copyright (c) 2023, lsl.
 ******************************************************************************/

#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>

void warp_matmul_v8(const torch::Tensor C, const torch::Tensor A, torch::Tensor B);
// C: row-major 
// A: row-major 
// B: col-major
void cute_matmul_v1(const torch::Tensor C, const torch::Tensor A, torch::Tensor B);

// void cute_matmul_v2(const torch::Tensor C, const torch::Tensor A, torch::Tensor B);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("cpp_warp_matmul_v8", &warp_matmul_v8, "lsl warp matmul v8");
  m.def("cpp_cute_matmul_v1", &cute_matmul_v1, "lsl cute matmul v1");
}