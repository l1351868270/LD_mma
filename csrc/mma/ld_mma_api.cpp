/******************************************************************************
 * Copyright (c) 2023, lsl.
 ******************************************************************************/

#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>

void bank_conflicts(int offset);

void cublas_matmul(const torch::Tensor C, const torch::Tensor A, torch::Tensor B);

void warp_matmul_v8(const torch::Tensor C, const torch::Tensor A, torch::Tensor B);

// C: row-major 
// A: row-major 
// B: col-major
void cute_matmul_v1(const torch::Tensor C, const torch::Tensor A, torch::Tensor B);

void cute_matmul_v2(const torch::Tensor C, const torch::Tensor A, torch::Tensor B);

void cute_matmul_v3(const torch::Tensor C, const torch::Tensor A, torch::Tensor B);



PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("cpp_bank_conflicts", &bank_conflicts, "ld_mma bank conflicts tests");
  m.def("cpp_cublas_matmul", &cublas_matmul, "ld_mma cublas matmul");
  m.def("cpp_warp_matmul_v8", &warp_matmul_v8, "ld_mma warp matmul v8");
  m.def("cpp_cute_matmul_v1", &cute_matmul_v1, "ld_mma cute matmul v1");
  m.def("cpp_cute_matmul_v2", &cute_matmul_v2, "ld_mma cute matmul v2");
  m.def("cpp_cute_matmul_v3", &cute_matmul_v3, "ld_mma cute matmul v3");
}