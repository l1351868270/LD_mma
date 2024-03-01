
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
  
  M = 64 * 3
  N = 64 * 5
  K = 64 * 7

  Shape: Stride
    A B C
      (192,448):(448,_1)
      (320,448):(448,_1)
      (192,320):(320,_1)
    gA gB gC
      (_64,_64,7):(448,_1,_64) -> (_64,_64,1,7):(448,_1,0,_64)
      (_64,_64,7):(448,_1,_64) -> (_64,_64,1,7):(448,_1,0,_64)
      (_64,_64):(320,_1)
    sA sB sC
      (_64,_64):(_64,_1)
      (_64,_64):(_64,_1)
      (_64,_64):(_64,_1)

    // global memory -> shared memory
    g2sgA g2sgB
      ((_8,_1),_4,_1,7):((_1,_0),7168,_0,_64) -> ((_8,_1),_4,1,_1,7):((_1,_0),7168,0,_0,_64)  // 7168 = 64 * 7 * 16
      ((_8,_1),_4,_1,7):((_1,_0),7168,_0,_64) -> ((_8,_1),_4,1,_1,7):((_1,_0),7168,0,_0,_64)
    g2ssA g2ssB
      ((_8,_1),_4,_1):((_1,_0),_1024,_0) // 1024 = 64 * 16
      ((_8,_1),_4,_1):((_1,_0),_1024,_0)

    // shared memory -> registers
    // use ldmatrix
    s2rsA s2rsB
      ((_8,_1),_1,(_2,_2)):((_1,_0),_0,(16,32)) -> ([_8,_1],[_1,_2],[1,_2])):([_1,_0],[_0,16],[0,32])
      ((_8,_1),_4,(_2,_2)):((_1,_0),_1024,(16,32)) -> ([_8,_1],[_4,1],[1,_2],[1,_2]):([_1,_0],[_1024,0],[0,16],[0,32]) // 1024 = 64 * 16

    s2rrA s2rrB rC
      ((_2,_2,_2),_1,(_2,_2)):((_1,_2,_4),_0,(_8,_16))
      ((_2,_2),_8,(_2,_2)):((_1,_2),_4,(_32,_64))
      ((_2,_2),_1,_8):((_1,_2),_0,_4)

    s2rrA_copy_view s2rrB_copy_view
      ((_8,_1),_1,_4):((_1,_0),_0,_8)
      ((_8,_1),_4,_4):((_1,_0),_8,_32)

    // registers -> global memory
    
    // registers -> shared memory
    r2srC
      ((_1,_8),_1,_4):((_0,_1),_0,_8)
    r2ssC
      ((_1,(_2,_2,_2)),_1,(_2,_2)):((_0,(_1,_512,8)),_0,(16,32))

    // shared memory -> global memory
    s2gsC
      ((_1,_8),_4,_1):((_0,_1),_1024,_0)
    s2ggC
      ((_1,_8),_4,_1):((_0,_1),5120,_0)
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
  using SmemCopyAtom = typename Cute_traits::SmemCopyAtom;
  using GmemTiledCopyAB = typename Cute_traits::GmemTiledCopyAB;
  using SmemCopyAtomC = typename Cute_traits::SmemCopyAtomC;
  using GmemTiledCopyC = typename Cute_traits::GmemTiledCopyC;
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

  if (cute::thread(debug_thread)) { 
    printf("A B C\n");
    cute::print(A.layout()); 
    printf("\n"); 
    cute::print(B.layout()); 
    printf("\n");
    cute::print(C.layout()); 
    printf("\n");
  }

  cute::Tensor gA = cute::local_tile(A, cute::make_tile(cute::Int<kTile_M>{}, cute::Int<kTile_K>{}), cute::make_coord(Tile_m, cute::_));
  cute::Tensor gB = cute::local_tile(B, cute::make_tile(cute::Int<kTile_N>{}, cute::Int<kTile_K>{}), cute::make_coord(Tile_n, cute::_));
  cute::Tensor gC = cute::local_tile(C, cute::make_tile(cute::Int<kTile_M>{}, cute::Int<kTile_N>{}), cute::make_coord(Tile_m, Tile_n));

  if (cute::thread(debug_thread)) { 
    printf("gA gB gC\n");
    cute::print(gA.layout()); 
    printf("\n"); 
    cute::print(gB.layout()); 
    printf("\n");
    cute::print(gC.layout()); 
    printf("\n");
    cute::print(cute::size<0>(gA)); 
    printf("\n");
    cute::print(cute::size<1>(gA)); 
    printf("\n");
    cute::print(cute::size<2>(gA)); 
    printf("\n");
  }

  auto sA = make_tensor(cute::make_smem_ptr(smem_a), SmemLayoutA{});
  auto sB = make_tensor(cute::make_smem_ptr(smem_b), SmemLayoutB{});
  auto sC = make_tensor(cute::make_smem_ptr(smem_c), SmemLayoutC{});
  
  if (cute::thread(debug_thread)) { 
    printf("sA sB sC\n");
    cute::print(sA.layout()); 
    printf("\n"); 
    cute::print(sB.layout()); 
    printf("\n");
    cute::print(sC.layout()); 
    printf("\n");
  }

  // global memory -> shared memory
  GmemTiledCopyAB g2s_tiled_copy_AB;
  auto g2s_thr_copy_AB = g2s_tiled_copy_AB.get_thread_slice(tidx);

  cute::Tensor g2sgA = g2s_thr_copy_AB.partition_S(gA);
  cute::Tensor g2sgB = g2s_thr_copy_AB.partition_S(gB);

  if (cute::thread(debug_thread)) { 
    printf("g2sgA g2sgB\n");
    cute::print(g2sgA.layout()); 
    printf("\n"); 
    cute::print(g2sgB.layout()); 
    printf("\n");
  }

  cute::Tensor g2ssA = g2s_thr_copy_AB.partition_D(sA);
  cute::Tensor g2ssB = g2s_thr_copy_AB.partition_D(sB);

  if (cute::thread(debug_thread)) { 
    printf("g2ssA g2ssB\n");
    cute::print(g2ssA.layout()); 
    printf("\n"); 
    cute::print(g2ssB.layout()); 
    printf("\n");
    cute::print(cute::rank(g2sgA)); 
    printf("\n");
    cute::print(cute::size<0>(g2sgA)); 
    printf("\n");
    cute::print(cute::size<1>(g2sgA)); 
    printf("\n");
    cute::print(cute::size<2>(g2sgA)); 
    printf("\n");
    cute::print(cute::rank(g2ssA)); 
    printf("\n");
    cute::print(cute::size<0>(g2ssA)); 
    printf("\n");
    cute::print(cute::size<1>(g2ssA)); 
    printf("\n");
    cute::print(cute::size<2>(g2ssA)); 
    printf("\n");
  }

  // shared memory -> registers
  TiledMma tiled_mma;
  auto s2r_tiled_copy_A = cute::make_tiled_copy_A(SmemCopyAtom{}, tiled_mma);
  auto s2r_thr_copy_A = s2r_tiled_copy_A.get_thread_slice(tidx);
  cute::Tensor s2rsA = s2r_thr_copy_A.partition_S(sA);

  auto s2r_tiled_copy_B = cute::make_tiled_copy_B(SmemCopyAtom{}, tiled_mma);
  auto s2r_thr_copy_B = s2r_tiled_copy_B.get_thread_slice(tidx);
  cute::Tensor s2rsB = s2r_thr_copy_B.partition_S(sB);

  if (cute::thread(debug_thread)) {
    printf("s2rsA s2rsB\n");
    cute::print(s2rsA.layout());
    printf("\n"); 
    cute::print(s2rsB.layout());
    printf("\n"); 
  }

  auto thr_mma = tiled_mma.get_thread_slice(tidx);
  cute::Tensor s2rrA  = thr_mma.partition_fragment_A(sA);                           // (MMA,MMA_M,MMA_K)
  cute::Tensor s2rrB  = thr_mma.partition_fragment_B(sB); 
  cute::Tensor rC = cute::partition_fragment_C(tiled_mma, cute::Shape<cute::Int<kTile_M>, cute::Int<kTile_N>>{}); 

  if (cute::thread(debug_thread)) { 
    printf("s2rrA s2rrB rC\n");
    cute::print(s2rrA.layout()); 
    printf("\n"); 
    cute::print(cute::rank(s2rrA)); 
    printf("\n");
    cute::print(cute::size<0>(s2rrA)); 
    printf("\n");
    cute::print(cute::size<1>(s2rrA)); 
    printf("\n");
    cute::print(cute::size<2>(s2rrA)); 
    printf("\n");
    cute::print(s2rrB.layout()); 
    printf("\n");
    cute::print(cute::rank(s2rrB)); 
    printf("\n");
    cute::print(cute::size<0>(s2rrB)); 
    printf("\n");
    cute::print(cute::size<1>(s2rrB)); 
    printf("\n");
    cute::print(cute::size<2>(s2rrB)); 
    printf("\n");
    cute::print(rC.layout()); 
    printf("\n");
  }

  int num_tile_k = cute::size<2>(gA);
//   if (cute::thread(debug_thread)) { 
//     printf("num_tile_k\n");
//     cute::print(cute::size<2>(gA)); 
//     printf("\n");
//   }
  clear(rC);
  for (int i = 0; i < num_tile_k; i++) {
    // global memory -> shared memory
    // for (int m = 0; m < cute::size<1>(g2ssA); ++m) {
    //     cute::copy(g2s_tiled_copy_AB, g2sgA(cute::_, m, cute::_, i), g2ssA(cute::_, m, cute::_));
    // }
    cute::copy(g2s_tiled_copy_AB, g2sgA(cute::_, cute::_, cute::_, i), g2ssA);
    cute::copy(g2s_tiled_copy_AB, g2sgB(cute::_, cute::_, cute::_, i), g2ssB);
    cute::cp_async_fence();
    cute::cp_async_wait<0>();
    __syncthreads();

    // shared memory -> registers
    cute::Tensor s2rrA_copy_view = s2r_thr_copy_A.retile_D(s2rrA);
    cute::Tensor s2rrB_copy_view = s2r_thr_copy_B.retile_D(s2rrB);

    if (cute::thread(debug_thread) && i == 0) {
      printf("s2rrA_copy_view s2rrB_copy_view\n");
      cute::print(s2rrA_copy_view.layout());
      printf("\n"); 
      cute::print(s2rrB_copy_view.layout());
      printf("\n"); 
    }

    cute::copy(s2r_tiled_copy_A, s2rsA(cute::_, cute::_, 0), s2rrA_copy_view(cute::_, cute::_, 0));
    cute::copy(s2r_tiled_copy_B, s2rsB(cute::_, cute::_, 0), s2rrB_copy_view(cute::_, cute::_, 0));

    for (int j = 0; j < cute::size<2>(s2rrA); ++j) {
        if ( j < cute::size<2>(s2rrA) - 1) {
          cute::copy(s2r_tiled_copy_A, s2rsA(cute::_, cute::_, j+1), s2rrA_copy_view(cute::_, cute::_, j+1));
          cute::copy(s2r_tiled_copy_B, s2rsB(cute::_, cute::_, j+1), s2rrB_copy_view(cute::_, cute::_, j+1));
        }

        cute::gemm(tiled_mma, s2rrA(cute::_, cute::_, j), s2rrB(cute::_, cute::_, j), rC);
    }
  }

  __syncthreads();
  
  // refer to https://github.com/Dao-AILab/flash-attention/blob/main/csrc/flash_attn/src/flash_fwd_kernel.h#L406
  // registers -> shared memory
  auto r2s_tiled_copy_C = make_tiled_copy_C(SmemCopyAtomC{}, tiled_mma);
  auto r2s_thr_copy_C = r2s_tiled_copy_C.get_thread_slice(tidx);
  cute::Tensor r2srC = r2s_thr_copy_C.retile_S(rC);
  cute::Tensor r2ssC = r2s_thr_copy_C.partition_D(sC);

  if (cute::thread(debug_thread)) { 
    printf("r2srC\n");
    cute::print(cute::layout(r2srC)); 
    printf("\n");
  }
  
  if (cute::thread(debug_thread)) { 
    printf("r2ssC\n");
    cute::print(cute::layout(r2ssC)); 
    printf("\n");
  }

  cute::copy(r2s_tiled_copy_C, r2srC, r2ssC);
  __syncthreads();

  // shared memory -> global memory
  GmemTiledCopyC s2g_tiled_copy_C;
  auto s2g_thr_copy_C = s2g_tiled_copy_C.get_thread_slice(tidx);
  cute::Tensor s2gsC = s2g_thr_copy_C.partition_S(sC);
  cute::Tensor s2ggC = s2g_thr_copy_C.partition_D(gC);

  if (cute::thread(debug_thread)) { 
    printf("s2gsC\n");
    cute::print(cute::layout(s2gsC)); 
    printf("\n");
  }

  if (cute::thread(debug_thread)) { 
    printf("s2ggC\n");
    cute::print(cute::layout(s2ggC)); 
    printf("\n");
  }

  __syncthreads();
  // when block.x and block.y is big, it will generate bank conflict. i also test the cublas, it also has bank conflict
  cute::copy(s2g_tiled_copy_C, s2gsC, s2ggC);
//   int tidRow = tidx / 8;
//   int tidCol = tidx % 8;
//   for (int i = 0; i < 4; i++) {
//     *(reinterpret_cast<uint4*>(reinterpret_cast<elem_type*>(Cptr) + Tile_m * kTile_M * N + Tile_n * kTile_N + i * 16 * N + + tidRow * N + tidCol * 8)) = *(reinterpret_cast<uint4*>(smem_ + i * 16 * kTile_N  + tidRow * kTile_N + tidCol * 8));
//   }
//   printf("tidx=%d, tCsC ptr=%p; tCgC=%p\n", tidx, tCsC.data(), tCgC.data());
//   __syncthreads();

    // int kTiledM = 64;
    // int kTiledN = 64;

    // extern __shared__ half sharedMem[];

    // int bidx = blockIdx.x;
    // int bidy = blockIdx.y;
    // int tidx = threadIdx.x;
    
    // int tidRow = tidx / 8;
    // int tidCol = tidx % 8;

    // // for (int i = 0; i < 4; i++)
    // // {
    // //     *(reinterpret_cast<uint4*>(smem_c + i * 16 * kTile_N  + tidRow * kTile_N + tidCol * 8)) = make_uint4(0, 0, 0, 0);
    // // }

    // // // *(reinterpret_cast<uint4*>(C + goffset_c)) = *(reinterpret_cast<uint4*>(smem_acc + ld_offset_c));
    // // index = tidx * offset;
    // for (int i = 0; i < 4; i++)
    // {
    //     *(reinterpret_cast<uint4*>(reinterpret_cast<half*>(Cptr) + Tile_m * kTile_M * N + Tile_n * kTile_N + i * 16 * N + + tidRow * N + tidCol * 8)) = *(reinterpret_cast<uint4*>(smem_c + i * 16 * kTile_N  + tidRow * kTile_N + tidCol * 8));
    // }
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
  cudaDeviceSynchronize();
  cudaPeekAtLastError();
}