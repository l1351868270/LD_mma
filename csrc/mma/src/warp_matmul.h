/*
16by8by16
Load A inputs from memory: 16Bytes per thread * 32 = 512 Bytes
Load B inputs from memory: 8Bytes per thread * 32 = 256 Bytes
Load A+B inputs from memory: 1024 + 512 = 768Bytes
Perform one Tensor Core operation: 2048 flops per warp = 16*8*16*2=4096flops
4096/768 = 5.333 flops/byte

RTX 3090
Peak FP16 Tensor TFLOPS with FP16 Accumulate =   142/284 TFLOP/s = 142/284 * 10 ^ 12 FLOPS/s = 
Memory Bandwidth = 936 GB/sec = 936 * 1024 * 1024 * 1024 Bytes/s = 1,005,022,347,264 Bytes/s 
计算密度: 141.29 flops/byte
5.333 * 936 GB/sec = 4096 / 768 * 1,005,022,347,264 = 5,360,119,185,408 FLOPS/s

5,360,119,185,408 / (142 * 10 ^ 12) = 0.037747 = 3.7%
*/

#include "warp_traits.h"
// #include "debug.h"
#include <mma.h>
#include <torch/python.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/cuda/Loops.cuh>

extern "C" __device__ uint32_t __nvvm_get_smem_pointer(void *ptr);

#define DEBUG_WARP_MATMUL_DEVICE 1

template <typename Warp_traits>
__global__ void WarpMatmulV8(void *Cptr, void *Aptr, void *Bptr, 
                                const int M, const int N, const int K)
{
  int Tile_m = blockIdx.x;
  int Tile_n = blockIdx.y;
  int tidx = threadIdx.x;

  constexpr static int kTile_M = Warp_traits::kTile_M;
  constexpr static int kTile_N = Warp_traits::kTile_N;
  constexpr static int kTile_K = Warp_traits::kTile_K;

  using elem_type = typename Warp_traits::elem_type;

  char* A = (char*)Aptr;
  char* B = (char*)Bptr;
  char* C = (char*)Cptr;

  int BYTES_PER_ELT = sizeof(elem_type);

  int ELTS_PER_THREADS = 8;

  int gstride_a[2] = {K, 1};
  int gstride_b[2] = {N, 1};
  int gstride_c[2] = {N, 1};

  int goffset_a = 0;
  int goffset_b = 0;
  int goffset_c = 0;

  int threads_layout_a[2] = {kTile_M, kTile_K / ELTS_PER_THREADS};
  int threads_layout_b[2] = {kTile_K, kTile_N / ELTS_PER_THREADS};
  int threads_layout_c[2] = {kTile_M, kTile_N / ELTS_PER_THREADS};

  int thread_coordinate_a[2] = {tidx / threads_layout_a[1], tidx % threads_layout_a[1]};
  int thread_coordinate_b[2] = {tidx / threads_layout_b[1], tidx % threads_layout_b[1]};
  int thread_coordinate_c[2] = {tidx / threads_layout_c[1], tidx % threads_layout_c[1]};

  int sstride_a[2] = {kTile_K, 1};
  int sstride_b[2] = {kTile_N, 1};
  int sstride_c[2] = {kTile_N, 1};

  int soffset_a = 0;
  int soffset_b = 0;
  int soffset_c = 0;
  soffset_a += thread_coordinate_a[0] * sstride_a[0] * BYTES_PER_ELT;
  soffset_a += thread_coordinate_a[1] * sstride_a[1] * ELTS_PER_THREADS * BYTES_PER_ELT;
  soffset_b += thread_coordinate_b[0] * sstride_b[0] * BYTES_PER_ELT;
  soffset_b += thread_coordinate_b[1] * sstride_b[1] * ELTS_PER_THREADS * BYTES_PER_ELT;
  soffset_c += thread_coordinate_c[0] * sstride_c[0] * BYTES_PER_ELT;
  soffset_c += thread_coordinate_c[1] * sstride_c[1] * ELTS_PER_THREADS * BYTES_PER_ELT;

  int ld_offset_a = 0;
  int ld_offset_b = 0;
  int ld_offset_c = 0;

  int ld_thread_coordinate_a[2] = {tidx % 16, tidx / 16};
  int ld_thread_coordinate_b[2] = {tidx % 16, tidx / 16};
  int ld_thread_coordinate_c[2] = {tidx % 16, tidx / 16};

  ld_offset_a += ld_thread_coordinate_a[0] * sstride_a[0] * BYTES_PER_ELT;
  ld_offset_a += ld_thread_coordinate_a[1] * sstride_a[1] * ELTS_PER_THREADS * BYTES_PER_ELT;
  ld_offset_b += ld_thread_coordinate_b[0] * sstride_b[0] * BYTES_PER_ELT;
  ld_offset_b += ld_thread_coordinate_b[1] * sstride_b[1] * ELTS_PER_THREADS * BYTES_PER_ELT;
  ld_offset_c += ld_thread_coordinate_c[0] * sstride_c[0] * BYTES_PER_ELT;
  ld_offset_c += ld_thread_coordinate_c[1] * sstride_c[1] * ELTS_PER_THREADS * BYTES_PER_ELT;

  extern __shared__ char smem_[];
  char* smem_a = smem_;
  char* smem_b = smem_ + BYTES_PER_ELT * kTile_M * kTile_K; 
  char* smem_acc = smem_ + BYTES_PER_ELT * kTile_M * kTile_K + BYTES_PER_ELT * kTile_K * kTile_N; 
  // char* smem_tmp = smem_ + BYTES_PER_ELT * MMA_M * MMA_K + BYTES_PER_ELT * MMA_K * MMA_N + BYTES_PER_ELT * MMA_M * MMA_N; 

  int K_tiles = K / kTile_K;

  // // int laneid = lane_id();

  int groupID = tidx / 4;
  int threadID_in_group = tidx % 4;

  // #ifdef DEBUG_WARP_MATMUL_DEVICE
  //     if (tidx == 0) {
  //         printf("WarpMatmulV8: BYTES_PER_ELT K_tiles is: %d\n", K_tiles);
  //     }
  // #endif

  uint2 s_c = make_uint2(0, 0);
  for (int Tile_k = 0; Tile_k < K_tiles; Tile_k++) {
    // Load data from global memory to shared memory

    // 1. offset - global memory pointer to the Tile Block
    goffset_a = 0;
    goffset_b = 0;

    goffset_a += Tile_m * kTile_M * gstride_a[0] * BYTES_PER_ELT;
    goffset_a += Tile_k * kTile_K * gstride_a[1] * BYTES_PER_ELT;

    goffset_b += Tile_k * kTile_K * gstride_b[0] * BYTES_PER_ELT;
    goffset_b += Tile_n * kTile_N * gstride_b[1] * BYTES_PER_ELT;

    // 2. offset - global memory pointer to Thread Block
    // per thread load 8 elements of a row
    // thread

    goffset_a += thread_coordinate_a[0] * gstride_a[0] * BYTES_PER_ELT;
    goffset_a += thread_coordinate_a[1] * gstride_a[1] * ELTS_PER_THREADS * BYTES_PER_ELT;

    goffset_b += thread_coordinate_b[0] * gstride_b[0] * BYTES_PER_ELT;
    goffset_b += thread_coordinate_b[1] * gstride_b[1] * ELTS_PER_THREADS * BYTES_PER_ELT;

    uint4 dst_a = make_uint4(0, 0, 0, 0);
    
    if (thread_coordinate_a[0] < threads_layout_a[0]) {
      dst_a = *reinterpret_cast<const uint4*>(A + goffset_a);
      *reinterpret_cast<uint4*>(smem_a + soffset_a) = dst_a;
    }

    uint4 dst_b = make_uint4(0, 0, 0, 0); 
    
    if (thread_coordinate_b[0] < threads_layout_b[0]) {
      dst_b = *reinterpret_cast<const uint4*>(B + goffset_b);
      *reinterpret_cast<uint4*>(smem_b + soffset_b) = dst_b;
    }

    __syncthreads();

    uint4 s_a = make_uint4(0, 0, 0, 0);

    uint32_t tt_a = __nvvm_get_smem_pointer(smem_a + ld_offset_a);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(s_a.x), "=r"(s_a.y), "=r"(s_a.z), "=r"(s_a.w) 
        : "r"(tt_a));

    uint2 s_b = make_uint2(0, 0);

    
    uint32_t tt_b = __nvvm_get_smem_pointer(smem_b + ld_offset_b);
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0, %1}, [%2];\n"
        : "=r"(s_b.x), "=r"(s_b.y) 
        : "r"(tt_b));

    __syncthreads();

  //   asm volatile( \
  //           "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 \n" \
  //           "    {%0, %1}, \n" \
  //           "    {%2, %3, %4, %5}, \n" \
  //           "    {%6, %7}, \n" \
  //           "    {%8, %9}; \n" \
  //                   : "=r"(  s_c.x), "=r"(  s_c.y)
  //                   :  "r"(s_a.x),  "r"(s_a.y),  "r"(s_a.z),  "r"(s_a.w)
  //                   ,  "r"(s_b.x),  "r"(s_b.y)
  //                   ,  "r"(s_c.x),  "r"(s_c.y));
      asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
      "{%0,  %1},"
      "{%2,  %3,  %4,  %5},"
      "{%6,  %7},"
      "{%8,  %9};\n"
      : "=r"(s_c.x), "=r"(s_c.y)
      :  "r"(s_a.x),  "r"(s_a.y),  "r"(s_a.z),  "r"(s_a.w),
         "r"(s_b.x),  "r"(s_b.y),
         "r"(s_c.x),  "r"(s_c.y));
  }



  __syncthreads();
  // asm volatile ("stmatrix.sync.aligned.x2.m8n8.shared.b16 [%0], {%1, %2};\n"
  //       : "=r"(smem_acc + ld_offset_c),
  //          "r"(s_c.x), "r"(s_c.y));

  *(reinterpret_cast<uint32_t*>(smem_acc) + groupID * 4 + threadID_in_group) = s_c.x;
  *(reinterpret_cast<uint32_t*>(smem_acc) + 32 + groupID * 4 + threadID_in_group) = s_c.y;

  __syncthreads();

  goffset_c += Tile_m * kTile_M * gstride_c[0] * BYTES_PER_ELT;
  goffset_c += Tile_n * kTile_N * gstride_c[1] * BYTES_PER_ELT;

  goffset_c += thread_coordinate_c[0] * gstride_c[0] * BYTES_PER_ELT;
  goffset_c += thread_coordinate_c[1] * gstride_c[1] * ELTS_PER_THREADS * BYTES_PER_ELT;
  
  uint4 t_a = *(reinterpret_cast<uint4*>(C + goffset_c));
  if (thread_coordinate_c[0] < threads_layout_c[0]) {
    #ifdef DEBUG_WARP_MATMUL_DEVICE
          // printf("WarpMatmulV8: thread_coordinate_c: [%d, %d], goffset_c size is: %d, ld_offset_c is %d\n", 
          //         thread_coordinate_c[0], thread_coordinate_c[1], goffset_c, ld_offset_c);
          // __half sc_0 = *reinterpret_cast<__half*>(smem_acc + 4 * groupID *4 + 4* threadID_in_group);
          // __half sc_1 = *reinterpret_cast<__half*>(smem_acc + 4 * groupID *4 + 4* threadID_in_group + 2);
          // __half sc_2 = *reinterpret_cast<__half*>(smem_acc + 32 * 4 + 4 * groupID *4 + 4* threadID_in_group);
          // __half sc_3 = *reinterpret_cast<__half*>(smem_acc + 32 * 4 + 4 * groupID *4 + 4* threadID_in_group + 2);
          // printf("Tile_m: %d, Tile_n %d, thread_coordinate_c: [%d, %d], goffset_c size is: %d, ld_offset_c is %d, sc_0 %f sc_1 %f sc_2 %f sc_3 %f\n",
          //         Tile_m, Tile_n, thread_coordinate_c[0], thread_coordinate_c[1], goffset_c, ld_offset_c, __half2float(sc_0),  __half2float(sc_1), __half2float(sc_2), __half2float(sc_3));
    #endif
    *(reinterpret_cast<uint4*>(C + goffset_c)) = *(reinterpret_cast<uint4*>(smem_acc + ld_offset_c));
  }
}

template <typename Warp_traits>
void warp_matmul_v8_cuda(const torch::Tensor C, const torch::Tensor A, torch::Tensor B, const int M, const int N, const int K) {
  constexpr static int kTile_M = Warp_traits::kTile_M;
  constexpr static int kTile_N = Warp_traits::kTile_N;

  constexpr static int Warp_M = Warp_traits::Warp_M;
  constexpr static int Warp_N = Warp_traits::Warp_N;
  constexpr static int Warp_K = Warp_traits::Warp_K;

  using elem_type = typename Warp_traits::elem_type;

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
                WarpMatmulV8<Warp_traits>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
  printf("threads is (%d, %d, %d); blocks is (%d, %d, %d)\n", threads.x, threads.y, threads.z, blocks.x, blocks.y, blocks.z);
  
  WarpMatmulV8<Warp_traits><<<blocks, threads, smem_size>>>(C_data, A_data, B_data, M, N, K);
  cudaPeekAtLastError();
  time_t t = time(NULL) - now;
  printf("cuda cost is %d\n", t);
}
