/******************************************************************************
 * Copyright (c) 2023, lsl.
 ******************************************************************************/

#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>

void matrix_v8(const torch::Tensor x1, const torch::Tensor x2, torch::Tensor out);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("lsl_matrix_v8", &matrix_v8, "lsl matrix v8");
}