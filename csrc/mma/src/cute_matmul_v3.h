
#include <mma.h>
#include <torch/python.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/cuda/Loops.cuh>

#include "cute_traits_v3.h"
#include "cute/algorithm/copy.hpp"

#include "cutlass/cutlass.h"
#include "cutlass/layout/layout.h"
#include <cutlass/numeric_types.h>

#define DEBUG_WARP_MATMUL_DEVICE 1

/*
  C: row-major 
  A: row-major 
  B: col-major
  
  M = 16 * 2 = 32
  N = 16 * 3 = 48
  K = 16 * 1024 = 16384

  Shape: Stride
    A B C
    (32,16384):(16384,_1)
    (48,16384):(16384,_1)
    (32,48):(48,_1)

    gA gB gC
      (_16,_16,1024):(16384,_1,_16) -> (_16,_16,1,1024):(16384,_1,16384*16,_16)
      (_16,_16,1024):(16384,_1,_16) -> (_16,_16,1,1024):(16384,_1,16384*16,_16)
      (_16,_16):(48,_1) -> (_16,_16,1,1):(48,_1,48*16,16)

    sA sB sC
      (_16,_16):(_16,_1)
      (_16,_16):(_16,_1)
      (_16,_16):(_16,_1)

    // global memory -> shared memory
    tAgA tBgB
      ((_8,_1),_1,_1,1024):((_1,_0),_0,_0,_16) -> ((_8,_1),1,_1,_1,1024):((_1,_0),0,_0,_0,_16)
      ((_8,_1),_1,_1,1024):((_1,_0),_0,_0,_16) -> ((_8,_1),1,_1,_1,1024):((_1,_0),0,_0,_0,_16)

    tAsA tBsB
      ((_8,_1),_1,_1):((_1,_0),_0,_0) 
      ((_8,_1),_1,_1):((_1,_0),_0,_0)

    // shared memory -> registers
    tSsA tSsB
      ((_8,_1),_1,_1):((_1,_0),_0,_0)
      ((_8,_1),_1,_1):((_1,_0),_0,_0)

    tSrA tSrB tSrC
      ((_2,_2,_2),_1,_1):((_1,_2,_4),_0,_0) -> ([1,_2],[_2,_2],[_1,_1]):([0,_1],[_2,_4],[_0,_0])
      ((_2,_2),_2,_1):((_1,_2),_4,_0) -> ([1,_2],[1,_2],[_2,_1]):([0,_1],[0,_2],[_4,_0])
      ((_2,_2),_1,_2):((_1,_2),_0,_4) -> ([1,_2],[_2,1],[_1,_2]):([0,_1],[_2,0],[_0,_4])

    // registers -> global memory
    tCgC
      ((_2,_2),_1,_2):((_1,384),_0,_8) -> ([1,_2],[_2,1],[_1,_2]):([0,_1],[384,0],[_0,_8])
    tSrC_copy_view
      ((_8,_1),_1,_1):((_1,_0),_0,_0)
*/
template <typename Cute_traits>
__global__ void CuteMatmulV3(void *Cptr, void *Aptr, void *Bptr, 
                                const int M, const int N, const int K)
{
  int Tile_m = blockIdx.x;
  int Tile_n = blockIdx.y;
  int tidx = threadIdx.x;

  int debug_thread = 0;

  constexpr static int kTile_M = Cute_traits::kTile_M;
  constexpr static int kTile_N = Cute_traits::kTile_N;
  constexpr static int kTile_K = Cute_traits::kTile_K;

  using elem_type = typename Cute_traits::Element;

  using SmemLayoutA = typename Cute_traits::SmemLayoutA;
  using SmemLayoutB = typename Cute_traits::SmemLayoutB;
  using SmemLayoutC = typename Cute_traits::SmemLayoutC;
  using GmemTiledCopyAB = typename Cute_traits::GmemTiledCopyAB;
  using SmemCopyAtom = typename Cute_traits::SmemCopyAtom;
  using TiledMma = typename Cute_traits::TiledMma;

  extern __shared__ elem_type smem_[];
  elem_type *smem_a = smem_;
  elem_type *smem_b = smem_ + cute::cosize(SmemLayoutA{});
  elem_type *smem_c = smem_ + cute::cosize(SmemLayoutA{}) + cute::cosize(SmemLayoutB{});

//   if (cute::thread(debug_thread)) {
//     printf("SmemLayoutA SmemLayoutB\n");
//     cute::print(cute::cosize(SmemLayoutA{}));
//     printf("\n");
//     cute::print(cute::cosize(SmemLayoutB{}));
//     printf("\n");
//     printf("smem_a %p\n", smem_a);
//     printf("smem_b %p\n", smem_b);
//     printf("smem_c %p\n", smem_c);
//   }

  cute::Tensor A = cute::make_tensor(cute::make_gmem_ptr(reinterpret_cast<elem_type*>(Aptr)), cute::make_shape(M, K), cute::make_stride(K, cute::Int<1>{}));
  cute::Tensor B = cute::make_tensor(cute::make_gmem_ptr(reinterpret_cast<elem_type*>(Bptr)), cute::make_shape(N, K), cute::make_stride(K, cute::Int<1>{}));
  cute::Tensor C = cute::make_tensor(cute::make_gmem_ptr(reinterpret_cast<elem_type*>(Cptr)), cute::make_shape(M, N), cute::make_stride(N, cute::Int<1>{}));

//   if (cute::thread(debug_thread)) { 
//     printf("A B C\n");
//     cute::print(A.layout()); 
//     printf("\n"); 
//     cute::print(B.layout()); 
//     printf("\n");
//     cute::print(C.layout()); 
//     printf("\n");
//   }

  cute::Tensor gA = cute::local_tile(A, cute::make_tile(cute::Int<kTile_M>{}, cute::Int<kTile_K>{}), cute::make_coord(Tile_m, cute::_));
  cute::Tensor gB = cute::local_tile(B, cute::make_tile(cute::Int<kTile_N>{}, cute::Int<kTile_K>{}), cute::make_coord(Tile_n, cute::_));
  cute::Tensor gC = cute::local_tile(C, cute::make_tile(cute::Int<kTile_M>{}, cute::Int<kTile_N>{}), cute::make_coord(Tile_m, Tile_n));

//   if (cute::thread(debug_thread)) { 
//     printf("gA gB gC\n");
//     cute::print(gA.layout()); 
//     printf("\n"); 
//     cute::print(gB.layout()); 
//     printf("\n");
//     cute::print(gC.layout()); 
//     printf("\n");
//     cute::print(cute::size<0>(gA)); 
//     printf("\n");
//     cute::print(cute::size<1>(gA)); 
//     printf("\n");
//     cute::print(cute::size<2>(gA)); 
//     printf("\n");
//   }

  auto sA = make_tensor(cute::make_smem_ptr(smem_a), SmemLayoutA{});
  auto sB = make_tensor(cute::make_smem_ptr(smem_b), SmemLayoutB{});
  auto sC = make_tensor(cute::make_smem_ptr(smem_c), SmemLayoutC{});
  
//   if (cute::thread(debug_thread)) { 
//     printf("sA sB sC\n");
//     cute::print(sA.layout()); 
//     printf("\n"); 
//     cute::print(sB.layout()); 
//     printf("\n");
//     cute::print(sC.layout()); 
//     printf("\n");
//   }

  // global memory -> shared memory
  GmemTiledCopyAB gmem_tiled_copy_AB;
  auto gmem_thr_copy_AB = gmem_tiled_copy_AB.get_thread_slice(tidx);

  cute::Tensor tAgA = gmem_thr_copy_AB.partition_S(gA);
  cute::Tensor tBgB = gmem_thr_copy_AB.partition_S(gB);

//   if (cute::thread(debug_thread)) { 
//     printf("tAgA tBgB\n");
//     cute::print(tAgA.layout()); 
//     printf("\n"); 
//     cute::print(tBgB.layout()); 
//     printf("\n");
//   }

  cute::Tensor tAsA = gmem_thr_copy_AB.partition_D(sA);
  cute::Tensor tBsB = gmem_thr_copy_AB.partition_D(sB);

//   if (cute::thread(debug_thread)) { 
//     printf("tAsA tBsB\n");
//     cute::print(tAsA.layout()); 
//     printf("\n"); 
//     cute::print(tBsB.layout()); 
//     printf("\n");
//     cute::print(cute::rank(tAgA)); 
//     printf("\n");
//     cute::print(cute::size<0>(tAgA)); 
//     printf("\n");
//     cute::print(cute::size<1>(tAgA)); 
//     printf("\n");
//     cute::print(cute::size<2>(tAgA)); 
//     printf("\n");
//     cute::print(cute::rank(tAsA)); 
//     printf("\n");
//     cute::print(cute::size<0>(tAsA)); 
//     printf("\n");
//     cute::print(cute::size<1>(tAsA)); 
//     printf("\n");
//     cute::print(cute::size<2>(tAsA)); 
//     printf("\n");
//   }

  // shared memory -> registers
  TiledMma tiled_mma;
  auto smem_tiled_copy_A = cute::make_tiled_copy_A(SmemCopyAtom{}, tiled_mma);
  auto smem_thr_copy_A = smem_tiled_copy_A.get_thread_slice(tidx);
  cute::Tensor tSsA = smem_thr_copy_A.partition_S(sA);

  auto smem_tiled_copy_B = cute::make_tiled_copy_B(SmemCopyAtom{}, tiled_mma);
  auto smem_thr_copy_B = smem_tiled_copy_B.get_thread_slice(tidx);
  cute::Tensor tSsB = smem_thr_copy_B.partition_S(sB);

//   if (cute::thread(debug_thread)) {
//     printf("tSsA tSsB\n");
//     cute::print(tSsA.layout());
//     printf("\n"); 
//     cute::print(tSsB.layout());
//     printf("\n"); 
//   }

  auto thr_mma = tiled_mma.get_thread_slice(tidx);
  cute::Tensor tSrA  = thr_mma.partition_fragment_A(sA);                           // (MMA,MMA_M,MMA_K)
  cute::Tensor tSrB  = thr_mma.partition_fragment_B(sB); 
  cute::Tensor tSrC = cute::partition_fragment_C(tiled_mma, cute::Shape<cute::Int<kTile_M>, cute::Int<kTile_N>>{}); 

//   if (cute::thread(debug_thread)) { 
//     printf("tSrA tSrB tSrC\n");
//     cute::print(tSrA.layout()); 
//     printf("\n"); 
//     cute::print(cute::rank(tSrA)); 
//     printf("\n");
//     cute::print(cute::size<0>(tSrA)); 
//     printf("\n");
//     cute::print(cute::size<1>(tSrA)); 
//     printf("\n");
//     cute::print(cute::size<2>(tSrA)); 
//     printf("\n");
//     cute::print(tSrB.layout()); 
//     printf("\n");
//     cute::print(cute::rank(tSrB)); 
//     printf("\n");
//     cute::print(cute::size<0>(tSrB)); 
//     printf("\n");
//     cute::print(cute::size<1>(tSrB)); 
//     printf("\n");
//     cute::print(cute::size<2>(tSrB)); 
//     printf("\n");
//     cute::print(tSrC.layout()); 
//     printf("\n");
//   }

  int num_tile_k = cute::size<2>(gA);
//   if (cute::thread(debug_thread)) { 
//     printf("num_tile_k\n");
//     cute::print(cute::size<2>(gA)); 
//     printf("\n");
//   }
  clear(tSrC);
  for (int i = 0; i < num_tile_k; i++) {
    // global memory -> shared memory

    // for (int m = 0; m < cute::size<1>(tAsA); ++m) {
    //     cute::copy(gmem_tiled_copy_AB, tAgA(cute::_, m, cute::_, i), tAsA(cute::_, m, cute::_));
    // }
    cute::copy(gmem_tiled_copy_AB, tAgA(cute::_, cute::_, cute::_, i), tAsA);
    cute::copy(gmem_tiled_copy_AB, tBgB(cute::_, cute::_, cute::_, i), tBsB);
    cute::cp_async_fence();
    cute::cp_async_wait<0>();
    __syncthreads();

    // shared memory -> registers
    cute::Tensor tSrA_copy_view = smem_thr_copy_A.retile_D(tSrA);
    cute::Tensor tSrB_copy_view = smem_thr_copy_B.retile_D(tSrB);

    for (int j = 0; j < cute::size<2>(tSrA); ++j) {
        cute::copy(smem_tiled_copy_A, tSsA(cute::_, cute::_, j), tSrA_copy_view(cute::_, cute::_, j));
        cute::copy(smem_tiled_copy_B, tSsB(cute::_, cute::_, j), tSrB_copy_view(cute::_, cute::_, j));
        cute::gemm(tiled_mma, tSrA(cute::_, cute::_, j), tSrB(cute::_, cute::_, j), tSrC);
    }
  }

  __syncthreads();

//   if (cute::thread(debug_thread)) { 
//     printf("sA sB\n");
//     cute::print_tensor(sA); 
//     printf("\n");
//   }

  // registers -> global memory
  auto tCgC = thr_mma.partition_C(gC);

//   if (cute::thread(debug_thread)) { 
//     printf("tCgC\n");
//     cute::print(tCgC.layout()); 
//     printf("\n");
//   }

  cute::Tensor tSrC_copy_view = smem_thr_copy_A.retile_D(tSrC);

//   if (cute::thread(debug_thread)) { 
//       printf("tSrC_copy_view\n");
//       cute::print(tSrC_copy_view.layout()); 
//       printf("\n");
//   }

  cute::copy(tSrC, tCgC); 
//   __syncthreads();
}

// C: row-major 
// A: row-major 
// B: col-major
template <typename Cute_traits>
void cute_matmul_v3_cuda(const torch::Tensor C, const torch::Tensor A, torch::Tensor B, const int M, const int N, const int K) {
  constexpr static int kTile_M = Cute_traits::kTile_M;
  constexpr static int kTile_N = Cute_traits::kTile_N;
  constexpr static int kTile_K = Cute_traits::kTile_K;

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
//   int smem_size = ((M * N + M * K + N * K) * sizeof(elem_type) + 1024 - 1) / 1024;
  int smem_size = (kTile_M * kTile_N + kTile_M * kTile_K + kTile_N * kTile_K) * sizeof(elem_type);
  cudaFuncSetAttribute(
                CuteMatmulV3<Cute_traits>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
  printf("threads is (%d, %d, %d); blocks is (%d, %d, %d);smem_size is %d\n", threads.x, threads.y, threads.z, blocks.x, blocks.y, blocks.z, smem_size);
  
  CuteMatmulV3<Cute_traits><<<blocks, threads, smem_size>>>(C_data, A_data, B_data, M, N, K);
  cudaPeekAtLastError();
}