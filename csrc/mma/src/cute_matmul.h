
#include <mma.h>
#include <torch/python.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/cuda/Loops.cuh>

#include "cute_traits.h"
#include "cute/algorithm/copy.hpp"

#include "cutlass/cutlass.h"
#include "cutlass/layout/layout.h"
#include <cutlass/numeric_types.h>

extern "C" __device__ uint32_t __nvvm_get_smem_pointer(void *ptr);

#define DEBUG_WARP_MATMUL_DEVICE 1


/*
  C: row-major 
  A: row-major 
  B: col-major
  
  M = 16 * 2 = 32
  N = 16 * 3 = 48
  K = 16 * 1024 = 16384

  print(Layout)
    Shape: Stride
    A: (32,16384):(16384,_1)
    B: (48,16384):(16384,_1)
    C: (32,48):(48,_1)

  make_coord(cute::_, cute::_)
    gA: (_16,_16,2,1024):(16384,_1,262144,_16)
    gB: (_16,_16,3,1024):(16384,_1,262144,_16)
    gC: (_16,_16,2,3):(48,_1,768,_16)

    tAgA: ((_2,_2,_2),_1,_1,2,1024):((_1,131072,_8),_0,_0,262144,_16) => (([1,_2],[_2,_2]),[_1,_1],[2,1024]):((0,_1,131072,_8),_0,_0,262144,_16)
    tBgB: ((_2,_2),_2,_1,3,1024):((_1,_8),131072,_0,262144,_16) => (([1,_2],[1,_2]),[_2,_1],[3,1024]):((0, _1, 0, _8),131072,_0,262144,_16)
    tCgC: ((_2,_2),_1,_2,2,3):((_1,384),_0,_8,768,_16) => (([1,_2],[_2,1]),[_1,_2],[2,3]):((0,_1,[384,0]),[_0,_8],[768,_16])

  make_coord(Tile_m, cute::_)
  make_coord(Tile_m, cute::_)
  make_coord(Tile_m, Tile_n)
    gA: (_16,_16,1024):(16384,_1,_16) => 
    gB: (_16,_16,1024):(16384,_1,_16)
    gC: (_16,_16):(48,_1)
    tAgA: ((_2,_2,_2),_1,_1,1024):((_1,131072,_8),_0,_0,_16)
    tBgB: ((_2,_2),_2,_1,1024):((_1,_8),131072,_0,_16)
    tCgC: ((_2,_2),_1,_2):((_1,384),_0,_8)
    tArA: ((_2,_2,_2),_1,_1):((_1,_2,_4),_0,_0) => (([1,_2],[_2,_2]),[_1,_1]):(([0,_1],[_2,_4]),[_0,_0])
    tBrB: ((_2,_2),_2,_1):((_1,_2),_4,_0) => (([1,_2],[1,_2]),[_2,_1]):(([0,_1],[0,_2]),[_4,_0])
    tCrC: ((_2,_2),_1,_2):((_1,_2),_0,_4) => (([1,_2],[_2,1]),[_1,_2]):(([0,_1],[_2,0]),[_0,_4])
*/
template <typename Cute_traits>
__global__ void CuteMatmulForwardV1(void *Cptr, void *Aptr, void *Bptr, 
                                const int M, const int N, const int K)
{
  int Tile_m = blockIdx.x;
  int Tile_n = blockIdx.y;
  int tidx = threadIdx.x;

  constexpr static int kTile_M = Cute_traits::kTile_M;
  constexpr static int kTile_N = Cute_traits::kTile_N;
  constexpr static int kTile_K = Cute_traits::kTile_K;

  using elem_type = typename Cute_traits::Element;

  cute::Tensor A = cute::make_tensor(cute::make_gmem_ptr(reinterpret_cast<elem_type*>(Aptr)), cute::make_shape(M, K), cute::make_stride(K, cute::Int<1>{}));
  cute::Tensor B = cute::make_tensor(cute::make_gmem_ptr(reinterpret_cast<elem_type*>(Bptr)), cute::make_shape(N, K), cute::make_stride(K, cute::Int<1>{}));
  cute::Tensor C = cute::make_tensor(cute::make_gmem_ptr(reinterpret_cast<float*>(Cptr)), cute::make_shape(M, N), cute::make_stride(N, cute::Int<1>{}));

  if (cute::thread0()) { 
    printf("A B C\n");
    cute::print(A.layout()); 
    printf("\n"); 
    cute::print(B.layout()); 
    printf("\n");
    cute::print(C.layout()); 
    printf("\n");
  }

  // cute::Tensor gA = cute::local_tile(A, cute::make_tile(cute::Int<kTile_M>{}, cute::Int<kTile_K>{}), cute::make_coord(cute::_, cute::_));
  // cute::Tensor gB = cute::local_tile(B, cute::make_tile(cute::Int<kTile_N>{}, cute::Int<kTile_K>{}), cute::make_coord(cute::_, cute::_));
  // cute::Tensor gC = cute::local_tile(C, cute::make_tile(cute::Int<kTile_M>{}, cute::Int<kTile_N>{}), cute::make_coord(cute::_, cute::_));
  
  cute::Tensor gA = cute::local_tile(A, cute::make_tile(cute::Int<kTile_M>{}, cute::Int<kTile_K>{}), cute::make_coord(Tile_m, cute::_));
  cute::Tensor gB = cute::local_tile(B, cute::make_tile(cute::Int<kTile_N>{}, cute::Int<kTile_K>{}), cute::make_coord(Tile_n, cute::_));
  cute::Tensor gC = cute::local_tile(C, cute::make_tile(cute::Int<kTile_M>{}, cute::Int<kTile_N>{}), cute::make_coord(Tile_m, Tile_n));

  if (cute::thread0()) { 
    printf("gA gB gC\n");
    cute::print(gA.layout()); 
    printf("\n"); 
    cute::print(gB.layout()); 
    printf("\n");
    cute::print(gC.layout()); 
    printf("\n");
  }

  typename Cute_traits::TiledMma tiled_mma;
  auto thr_mma = tiled_mma.get_slice(tidx);
  auto tAgA = thr_mma.partition_A(gA);
  auto tBgB = thr_mma.partition_B(gB);
  auto tCgC = thr_mma.partition_C(gC);

  if (cute::thread0()) { 
    printf("tAgA tBgB tCgC\n");
    cute::print(tAgA.layout()); 
    printf("\n"); 
    cute::print(tBgB.layout()); 
    printf("\n");
    cute::print(tCgC.layout()); 
    printf("\n");
  }

  auto tArA = thr_mma.partition_fragment_A(gA(cute::_, cute::_, 0));
  auto tBrB = thr_mma.partition_fragment_B(gB(cute::_, cute::_, 0));
  auto tCrC = thr_mma.partition_fragment_C(gC(cute::_, cute::_));

  if (cute::thread0()) { 
    printf("tArA tBrB tCrC\n");
    cute::print(tArA.layout()); 
    printf("\n"); 
    cute::print(tBrB.layout()); 
    printf("\n");
    cute::print(tCrC.layout()); 
    printf("\n");
  }

  cute::clear(tCrC);

  int num_tile_k = cute::size<2>(gA);
  for (int i = 0; i < num_tile_k; i++) {
    cute::copy(tAgA(cute::_, cute::_, cute::_, i), tArA);
    cute::copy(tBgB(cute::_, cute::_, cute::_, i), tBrB);
    cute::gemm(tiled_mma, tCrC, tArA, tBrB, tCrC);
  }
  cute::copy(tCrC, tCgC); 

}

// C: row-major 
// A: row-major 
// B: col-major
template <typename Cute_traits>
void cute_matmul_v1_cuda(const torch::Tensor C, const torch::Tensor A, torch::Tensor B, const int M, const int N, const int K) {
  constexpr static int kTile_M = Cute_traits::kTile_M;
  constexpr static int kTile_N = Cute_traits::kTile_N;

  constexpr static int Warp_M = Cute_traits::Warp_M;
  constexpr static int Warp_N = Cute_traits::Warp_N;
  constexpr static int Warp_K = Cute_traits::Warp_K;

  using elem_type = typename Cute_traits::Element;

  elem_type* A_data = (elem_type*)A.data_ptr();
  elem_type* B_data = (elem_type*)B.data_ptr();
  elem_type* C_data = (elem_type*)C.data_ptr();

  dim3 threads(32 * Warp_M * Warp_N * Warp_K);
  dim3 blocks((M + kTile_M - 1) / kTile_M, (N + kTile_N - 1) / kTile_N);

  // Launch the kernel
  time_t now = time(NULL);
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  std::cout << "sharedMemPerBlock: " << prop.sharedMemPerBlock 
            << " sharedMemPerMultiprocessor: " << prop.sharedMemPerMultiprocessor
            << " sharedMemPerBlockOptin: " << prop.sharedMemPerBlockOptin << std::endl;
  int smem_size = ((M * N + M * K + N * K) * sizeof(elem_type) + 1024 - 1) / 1024;
  cudaFuncSetAttribute(
                CuteMatmulForwardV1<Cute_traits>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
  printf("threads is (%d, %d, %d); blocks is (%d, %d, %d)\n", threads.x, threads.y, threads.z, blocks.x, blocks.y, blocks.z);
  
  CuteMatmulForwardV1<Cute_traits><<<blocks, threads, smem_size>>>(C_data, A_data, B_data, M, N, K);
  cudaPeekAtLastError();
  time_t t = time(NULL) - now;
  printf("cuda cost is %d\n", t);
}