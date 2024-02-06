
#include <mma.h>
#include <torch/python.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/cuda/Loops.cuh>

#include "cute_traits_v2.h"
#include "cute/algorithm/copy.hpp"

#include "cutlass/cutlass.h"
#include "cutlass/layout/layout.h"
#include <cutlass/numeric_types.h>

extern "C" __device__ uint32_t __nvvm_get_smem_pointer(void *ptr);

#define DEBUG_WARP_MATMUL_DEVICE 1

// C: row-major 
// A: row-major 
// B: col-major
template <typename Cute_traits>
__global__ void CuteMatmulV2(void *Cptr, void *Aptr, void *Bptr, 
                                const int M, const int N, const int K)
{
  int Tile_m = blockIdx.x;
  int Tile_n = blockIdx.y;
  int tidx = threadIdx.x;

  constexpr static int kTile_M = Cute_traits::kTile_M;
  constexpr static int kTile_N = Cute_traits::kTile_N;
  constexpr static int kTile_K = Cute_traits::kTile_K;

  using elem_type = typename Cute_traits::Element;

  using SmemLayoutA = typename Cute_traits::SmemLayoutA;
  using SmemLayoutB = typename Cute_traits::SmemLayoutB;
  using SmemLayoutC = typename Cute_traits::SmemLayoutC;
  using GmemTiledCopyAB = typename Cute_traits::GmemTiledCopyAB;
  using SmemCopyAtom = typename Cute_traits::SmemCopyAtom;

  extern __shared__ elem_type smem_[];
  elem_type *smem_a = smem_;
  elem_type *smem_b = smem_ + cute::cosize(SmemLayoutA{});
  elem_type *smem_c = smem_ + cute::cosize(SmemLayoutA{}) + cute::cosize(SmemLayoutB{});


  cute::Tensor A = cute::make_tensor(cute::make_gmem_ptr(reinterpret_cast<elem_type*>(Aptr)), cute::make_shape(M, K), cute::make_stride(K, cute::Int<1>{}));
  cute::Tensor B = cute::make_tensor(cute::make_gmem_ptr(reinterpret_cast<elem_type*>(Bptr)), cute::make_shape(N, K), cute::make_stride(K, cute::Int<1>{}));
  cute::Tensor C = cute::make_tensor(cute::make_gmem_ptr(reinterpret_cast<elem_type*>(Cptr)), cute::make_shape(M, N), cute::make_stride(N, cute::Int<1>{}));

  // if (cute::thread0()) { 
  //   printf("A B C\n");
  //   cute::print(A.layout()); 
  //   printf("\n"); 
  //   cute::print(B.layout()); 
  //   printf("\n");
  //   cute::print(C.layout()); 
  //   printf("\n");
  // }

  cute::Tensor gA = cute::local_tile(A, cute::make_tile(cute::Int<kTile_M>{}, cute::Int<kTile_K>{}), cute::make_coord(Tile_m, cute::_));
  cute::Tensor gB = cute::local_tile(B, cute::make_tile(cute::Int<kTile_N>{}, cute::Int<kTile_K>{}), cute::make_coord(Tile_n, cute::_));
  cute::Tensor gC = cute::local_tile(C, cute::make_tile(cute::Int<kTile_M>{}, cute::Int<kTile_N>{}), cute::make_coord(Tile_m, Tile_n));

  // if (cute::thread0()) { 
  //   printf("gA gB gC\n");
  //   cute::print(gA.layout()); 
  //   printf("\n"); 
  //   cute::print(gB.layout()); 
  //   printf("\n");
  //   cute::print(gC.layout()); 
  //   printf("\n");
  // }

  auto sA = make_tensor(cute::make_smem_ptr(smem_a), SmemLayoutA{});
  auto sB = make_tensor(cute::make_smem_ptr(smem_b), SmemLayoutB{});
  auto sC = make_tensor(cute::make_smem_ptr(smem_c), SmemLayoutC{});
  
  // if (cute::thread0()) { 
  //   printf("sA sB sC\n");
  //   cute::print(sA.layout()); 
  //   printf("\n"); 
  //   cute::print(sB.layout()); 
  //   printf("\n");
  //   cute::print(sC.layout()); 
  //   printf("\n");
  // }

  typename Cute_traits::TiledMma tiled_mma;
  auto thr_mma = tiled_mma.get_thread_slice(tidx);
  cute::Tensor tSrA  = thr_mma.partition_fragment_A(sA);                           // (MMA,MMA_M,MMA_K)
  cute::Tensor tSrB  = thr_mma.partition_fragment_B(sB); 
  // cute::Tensor tSrC  = thr_mma.partition_fragment_B(sC);
  cute::Tensor tSrC = cute::partition_fragment_C(tiled_mma, cute::Shape<cute::Int<kTile_M>, cute::Int<kTile_N>>{}); 
  // auto tAgA = thr_mma.partition_A(gA);
  // auto tBgB = thr_mma.partition_B(gB);
  auto tCgC = thr_mma.partition_C(gC);
  // if (cute::thread0()) { 
  //   printf("tSrA tSrB tSrC\n");
  //   cute::print(tSrA.layout()); 
  //   printf("\n"); 
  //   cute::print(tSrB.layout()); 
  //   printf("\n");
  //   cute::print(tSrC.layout()); 
  //   printf("\n");
  // }

  GmemTiledCopyAB gmem_tiled_copy_AB;
  auto gmem_thr_copy_AB = gmem_tiled_copy_AB.get_thread_slice(tidx);

  cute::Tensor tAgA = gmem_thr_copy_AB.partition_S(gA);
  cute::Tensor tBgB = gmem_thr_copy_AB.partition_S(gB);

  // if (cute::thread0()) { 
  //   printf("tAgA tBgB\n");
  //   cute::print(tAgA.layout()); 
  //   printf("\n"); 
  //   cute::print(tBgB.layout()); 
  //   printf("\n");
  // }

  cute::Tensor tAsA = gmem_thr_copy_AB.partition_D(sA);
  cute::Tensor tBsB = gmem_thr_copy_AB.partition_D(sB);

  // if (cute::thread0()) { 
  //   printf("tAsA tBsB\n");
  //   cute::print(tAsA.layout()); 
  //   printf("\n"); 
  //   cute::print(tBsB.layout()); 
  //   printf("\n");
  // }
  
  auto smem_tiled_copy_A = cute::make_tiled_copy_A(SmemCopyAtom{}, tiled_mma);
  auto smem_thr_copy_A = smem_tiled_copy_A.get_thread_slice(tidx);
  cute::Tensor tSsA = smem_thr_copy_A.partition_S(sA);

  auto smem_tiled_copy_B = cute::make_tiled_copy_B(SmemCopyAtom{}, tiled_mma);
  auto smem_thr_copy_B = smem_tiled_copy_B.get_thread_slice(tidx);
  cute::Tensor tSsB = smem_thr_copy_B.partition_S(sB);
  // if (cute::thread0()) {
  //   printf("tSsA tSsB\n");
  //   cute::print(tSsA.layout());
  //   printf("\n"); 
  //   cute::print(tSsB.layout());
  //   printf("\n"); 
  // }

  int num_tile_k = cute::size<2>(gA);
  clear(tSrC);
  for (int i = 0; i < num_tile_k; i++) {
    cute::copy(tAgA(cute::_, cute::_, cute::_, i), tAsA);
    cute::copy(tBgB(cute::_, cute::_, cute::_, i), tBsB);
    cute::cp_async_fence();
    cute::cp_async_wait<0>();
    __syncthreads();

    // for (int bid = 0; bid < gridDim.x*gridDim.y; bid++){
    //   if (cute::thread(0, bid)) {
    //   // __half A0 = 16.0;
    //   // printf("--------------------------tAsA[%d], %f\n", i, __half2float(A0));
    //     for (int i = 0; i < cute::cosize(SmemLayoutA{}); i++) {
    //       __half tmp = *(tAsA.data() + i);
    //       printf("bid=%d, m=%d, n=%d, tAsA[%d], %f\n", bid, Tile_m, Tile_n, i, __half2float(tmp));
    //     }
    //   }
    // }

    // if (cute::thread(31)) { 
    //   printf("num_tile_k tidx=%d i = %d tAsA, tBsB\n", tidx, i);
    //   cute::print_tensor(tAsA); 
    //   printf("\n");
    // }

    cute::Tensor tSrA_copy_view = smem_thr_copy_A.retile_D(tSrA);
    cute::Tensor tSrB_copy_view = smem_thr_copy_B.retile_D(tSrB);

    cute::copy(smem_tiled_copy_A, tSsA(cute::_, cute::_, cute::_0{}), tSrA_copy_view(cute::_, cute::_, cute::_0{}));
    cute::copy(smem_tiled_copy_B, tSsB(cute::_, cute::_, cute::_0{}), tSrB_copy_view(cute::_, cute::_, cute::_0{}));
    
    __syncthreads();

    // if (cute::thread(31)) { 
    //   printf("num_tile_k tidx=%d i = %d tSrA_copy_view, tSrB_copy_view\n", tidx, i);
    //   cute::print_tensor(tSrA_copy_view); 
    //   printf("\n");
    // }

    cute::gemm(tiled_mma, tSrA, tSrB, tSrC);
  }
  __syncthreads();

  cute::Tensor tSrC_copy_view = smem_thr_copy_A.retile_D(tSrC);
  // if (cute::thread(0)) { 
  //     printf("tidx=%d tSrC tCgC\n", tidx);
  //     cute::print(tSrC.layout()); 
  //     printf("\n");
  //     cute::print(tCgC.layout()); 
  //     printf("\n");
  // }

  
  cute::copy(tSrC, tCgC); 
  __syncthreads();
}

// C: row-major 
// A: row-major 
// B: col-major
template <typename Cute_traits>
void cute_matmul_v2_cuda(const torch::Tensor C, const torch::Tensor A, torch::Tensor B, const int M, const int N, const int K) {
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
                CuteMatmulV2<Cute_traits>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
  printf("threads is (%d, %d, %d); blocks is (%d, %d, %d)\n", threads.x, threads.y, threads.z, blocks.x, blocks.y, blocks.z);
  
  CuteMatmulV2<Cute_traits><<<blocks, threads, smem_size>>>(C_data, A_data, B_data, M, N, K);
  cudaPeekAtLastError();
}