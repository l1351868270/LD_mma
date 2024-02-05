
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>

#include <cublas_v2.h>

// C: row-major 
// A: row-major 
// B: col-major
void cublas_matmul(const torch::Tensor C, const torch::Tensor A, torch::Tensor B) {
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

    half* A_data = (half*)A.data_ptr();
    half* B_data = (half*)B.data_ptr();
    half* C_data = (half*)C.data_ptr();

    // cublas_matmul_cuda<Cute_traits_v2<kTile_M, kTile_N, kTile_K, Warp_M, Warp_N, Warp_K>>(C, A, B, M, N, K);
    cublasHandle_t handle;
    cublasCreate(&handle);
    half alpha = half(1.f);
    half beta = half(0.f);
    cublasStatus_t ret = cublasHgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
          	  N, M, K,
          	  &alpha,
          	  B_data, K,
          	  A_data, K,
          	  &beta,
          	  C_data, N);
}