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

#include <mma.h>
#include <torch/python.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/cuda/Loops.cuh>

extern "C" __device__ uint32_t __nvvm_get_smem_pointer(void *ptr);

using namespace nvcuda;

const int MMA_M = 16;
const int MMA_N = 8;
const int MMA_K = 16;

__global__
void MatrixForwardV8(half *a, half *b, half *c, int M, int N, int K)
{


  int ROW = blockIdx.x;
  int COL = blockIdx.y;

  int BYTES_PER_ELT = 2;
  int ELTS_PER_THREADS = 8;
  int BYTES_PER_THREADS = BYTES_PER_ELT * ELTS_PER_THREADS;

  extern __shared__ char smem_[];
  char* smem_a = smem_;
  char* smem_b = smem_ + BYTES_PER_ELT * MMA_M * MMA_K; 
  char* smem_acc = smem_ + BYTES_PER_ELT * MMA_M * MMA_K + BYTES_PER_ELT * MMA_K * MMA_N; 
  char* smem_tmp = smem_ + BYTES_PER_ELT * MMA_M * MMA_K + BYTES_PER_ELT * MMA_K * MMA_N + BYTES_PER_ELT * MMA_M * MMA_N; 

  int K_tiles = K / MMA_K;
  int ELTS_PER_ROW_a = K;
  int BYTES_PER_ROW_a = BYTES_PER_ELT * ELTS_PER_ROW_a;
  int TITLE_THREADS_PER_ROW_a = (BYTES_PER_ELT * MMA_K + BYTES_PER_THREADS - 1) / BYTES_PER_THREADS;

  int ELTS_PER_ROW_b = N;
  int BYTES_PER_ROW_b = BYTES_PER_ELT * ELTS_PER_ROW_b;
  int TITLE_THREADS_PER_ROW_b = (BYTES_PER_ELT * MMA_N + BYTES_PER_THREADS - 1) / BYTES_PER_THREADS;

  int tidx = threadIdx.x;

  // int laneid = lane_id();

  int groupID = tidx / 4;
  int threadID_in_group = tidx % 4;

  uint2 s_c = make_uint2(0, 0);

  for (int k = 0; k < K_tiles; k++) {
    int offset = 0;

    offset += ROW * MMA_M * ELTS_PER_ROW_a;
    offset += k * MMA_K;
    int row_a = tidx / TITLE_THREADS_PER_ROW_a;
    int col_a = tidx % TITLE_THREADS_PER_ROW_a;
    offset += row_a * ELTS_PER_ROW_a;
    offset += col_a * ELTS_PER_THREADS;

    uint4 dst_a = make_uint4(0, 0, 0, 0);
    
    if (row_a < MMA_M && col_a * ELTS_PER_THREADS < MMA_K) {
      dst_a = *reinterpret_cast<const uint4*>(a + offset);
      *reinterpret_cast<uint4*>(smem_a + row_a * MMA_K * BYTES_PER_ELT + col_a * ELTS_PER_THREADS * BYTES_PER_ELT) = dst_a;

    // __half sc_0 = *reinterpret_cast<__half*>(smem_a + row_a * MMA_K * BYTES_PER_ELT + col_a * ELTS_PER_THREADS * BYTES_PER_ELT);
    // __half sc_1 = *reinterpret_cast<__half*>(smem_a + row_a * MMA_K * BYTES_PER_ELT + col_a * ELTS_PER_THREADS * BYTES_PER_ELT + 2);
    // __half sc_2 = *reinterpret_cast<__half*>(smem_a + row_a * MMA_K * BYTES_PER_ELT + col_a * ELTS_PER_THREADS * BYTES_PER_ELT + 4);
    // __half sc_3 = *reinterpret_cast<__half*>(smem_a + row_a * MMA_K * BYTES_PER_ELT + col_a * ELTS_PER_THREADS * BYTES_PER_ELT + 6);
    // __half sc_4 = *reinterpret_cast<__half*>(smem_a + row_a * MMA_K * BYTES_PER_ELT + col_a * ELTS_PER_THREADS * BYTES_PER_ELT + 8);
    // __half sc_5 = *reinterpret_cast<__half*>(smem_a + row_a * MMA_K * BYTES_PER_ELT + col_a * ELTS_PER_THREADS * BYTES_PER_ELT + 10);
    // __half sc_6 = *reinterpret_cast<__half*>(smem_a + row_a * MMA_K * BYTES_PER_ELT + col_a * ELTS_PER_THREADS * BYTES_PER_ELT + 12);
    // __half sc_7 = *reinterpret_cast<__half*>(smem_a + row_a * MMA_K * BYTES_PER_ELT + col_a * ELTS_PER_THREADS * BYTES_PER_ELT + 14);

    //     printf("ROW: %d, COL %d, ELTS_PER_ROW_a %d THREADS_PER_ROW_a %d, row_a %d col_a %d offset %d sc_0 %f sc_1 %f sc_2 %f sc_3 %f sc_4 %f sc_5 %f sc_6 %f sc_7 %f\n",
    //       ROW, COL, ELTS_PER_ROW_a, TITLE_THREADS_PER_ROW_a, row_a, col_a, offset, __half2float(sc_0),  __half2float(sc_1),
    //        __half2float(sc_2), __half2float(sc_3), __half2float(sc_4), __half2float(sc_5), __half2float(sc_6), __half2float(sc_7));
    }

    offset = 0;
    offset += k * MMA_K * ELTS_PER_ROW_b;
    offset += COL * MMA_N ;
    int row_b = tidx / TITLE_THREADS_PER_ROW_b;
    int col_b = tidx % TITLE_THREADS_PER_ROW_b;
    offset += row_b * ELTS_PER_ROW_b;
    offset += col_b * ELTS_PER_THREADS;

    uint4 dst_b = make_uint4(0, 0, 0, 0); 
    
    if (row_b < MMA_K && col_b * ELTS_PER_THREADS < MMA_N) {
      dst_b = *reinterpret_cast<const uint4*>(&b[offset]);
      *reinterpret_cast<uint4*>(smem_b + row_b * MMA_N * BYTES_PER_ELT  + col_b * ELTS_PER_THREADS * BYTES_PER_ELT) = dst_b;
    }

    __syncthreads();

    uint4 s_a = make_uint4(0, 0, 0, 0);
    int ld_row_a = tidx % 16;
    int ld_col_a = tidx / 16;
    uint32_t tt_a = __nvvm_get_smem_pointer(smem_a + ld_row_a * MMA_K * BYTES_PER_ELT + 8 * ld_col_a * BYTES_PER_ELT);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(s_a.x), "=r"(s_a.y), "=r"(s_a.z), "=r"(s_a.w) 
        : "r"(tt_a));
        // : "r"(smem_a + row_a * MMA_K * BYTES_PER_ELT + col_a * ELTS_PER_THREADS * BYTES_PER_ELT));

    
    *(reinterpret_cast<uint32_t*>(smem_tmp + 4 * 8 * groupID + 4 * threadID_in_group)) = s_a.x;
    *(reinterpret_cast<uint32_t*>(smem_tmp + 4 * 8 * 8 + 4 * 8 * groupID + 4* threadID_in_group)) = s_a.y;
    *(reinterpret_cast<uint32_t*>(smem_tmp + 4 * 8 * groupID + 4 * 4 + 4 * threadID_in_group)) = s_a.z;
    *(reinterpret_cast<uint32_t*>(smem_tmp + 4 * 8 * 8 + 4 * 8 * groupID + 4 * 4 + 4 * threadID_in_group)) = s_a.w;

    // __half sc_0 = *reinterpret_cast<__half*>(smem_tmp + 4 * 8 * groupID + 4 * threadID_in_group);
    // __half sc_1 = *reinterpret_cast<__half*>(smem_tmp + 4 * 8 * groupID + 4 * threadID_in_group + 2);
    // __half sc_2 = *reinterpret_cast<__half*>(smem_tmp + 4 * 8 * 8 + 4 * 8 * groupID + 4* threadID_in_group);
    // __half sc_3 = *reinterpret_cast<__half*>(smem_tmp + 4 * 8 * 8 + 4 * 8 * groupID + 4* threadID_in_group + 2);
    
    // __half sc_4 = *reinterpret_cast<__half*>(smem_tmp + 4 * 8 * groupID + 4 * 4 + 4 * threadID_in_group);
    // __half sc_5 = *reinterpret_cast<__half*>(smem_tmp + 4 * 8 * groupID + 4 * 4 + 4 * threadID_in_group + 2);
    // __half sc_6 = *reinterpret_cast<__half*>(smem_tmp + 4 * 8 * 8 + 4 * 8 * groupID + 4 * 4 + 4 * threadID_in_group);
    // __half sc_7 = *reinterpret_cast<__half*>(smem_tmp + 4 * 8 * 8 + 4 * 8 * groupID + 4 * 4 + 4 * threadID_in_group + 2);

        // printf("ROW: %d, COL %d, ELTS_PER_ROW_a %d THREADS_PER_ROW_a %d, ld_row_a %d ld_col_a %d offset %d sc_0 %f sc_1 %f sc_2 %f sc_3 %f sc_4 %f sc_5 %f sc_6 %f sc_7 %f\n",
        //   ROW, COL, ELTS_PER_ROW_a, TITLE_THREADS_PER_ROW_a, ld_row_a, ld_col_a, offset, __half2float(sc_0),  __half2float(sc_1),
        //    __half2float(sc_2), __half2float(sc_3), __half2float(sc_4), __half2float(sc_5), __half2float(sc_6), __half2float(sc_7));

    uint2 s_b = make_uint2(0, 0);
    int ld_row_b = tidx % 16;
    int ld_col_b = tidx / 16;
    uint32_t tt_b = __nvvm_get_smem_pointer(smem_b + ld_row_b * MMA_N * BYTES_PER_ELT + ld_col_b * ELTS_PER_THREADS * BYTES_PER_ELT);
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0, %1}, [%2];\n"
        : "=r"(s_b.x), "=r"(s_b.y) 
        : "r"(tt_b));
        // : "r"(smem_b + row_b * MMA_K * BYTES_PER_ELT + col_b * ELTS_PER_THREADS * BYTES_PER_ELT));

    // *(reinterpret_cast<uint32_t*>(smem_tmp + 4 * 4 * groupID + 4 * threadID_in_group)) = s_b.x;
    // *(reinterpret_cast<uint32_t*>(smem_tmp + 4 * 4 * 8 + 4 * 4 * groupID + 4* threadID_in_group)) = s_b.y;

    // __half sc_0 = *reinterpret_cast<__half*>(smem_tmp + 4 * groupID *4 + 4* threadID_in_group);
    // __half sc_1 = *reinterpret_cast<__half*>(smem_tmp + 4 * groupID *4 + 4* threadID_in_group + 2);
    // __half sc_2 = *reinterpret_cast<__half*>(smem_tmp + 32 * 4 + 4 * groupID *4 + 4* threadID_in_group);
    // __half sc_3 = *reinterpret_cast<__half*>(smem_tmp + 32 * 4 + 4 * groupID *4 + 4* threadID_in_group + 2);

    // printf("ROW: %d, COL %d, ELTS_PER_ROW_b %d THREADS_PER_ROW_b %d, row_b %d col_b %d offset %d sc_0 %f sc_1 %f sc_2 %f sc_3 %f\n",
    //         ROW, COL, ELTS_PER_ROW_b, TITLE_THREADS_PER_ROW_b, row_b, col_b, offset, __half2float(sc_0),  __half2float(sc_1),
    //        __half2float(sc_2), __half2float(sc_3));

    // asm volatile( \
    //         "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 \n" \
    //         "    {%0, %1}, \n" \
    //         "    {%2, %3, %4, %5}, \n" \
    //         "    {%6, %7}, \n" \
    //         "    {%0, %1}; \n" \
    //                 : "=r"(  s_c.x), "=r"(  s_c.y)
    //                 :  "r"(s_a.x),  "r"(s_a.y),  "r"(s_a.z),  "r"(s_a.w)
    //                 ,  "r"(s_b.x),  "r"(s_b.y);

    asm volatile( \
            "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 \n" \
            "    {%0, %1}, \n" \
            "    {%2, %3, %4, %5}, \n" \
            "    {%6, %7}, \n" \
            "    {%8, %9}; \n" \
                    : "=r"(  s_c.x), "=r"(  s_c.y)
                    :  "r"(s_a.x),  "r"(s_a.y),  "r"(s_a.z),  "r"(s_a.w)
                    ,  "r"(s_b.x),  "r"(s_b.y)
                    ,  "r"(s_c.x),  "r"(s_c.y));

  }

  // char* 

  *(reinterpret_cast<uint32_t*>(smem_acc) + groupID * 4 + threadID_in_group) = s_c.x;
  *(reinterpret_cast<uint32_t*>(smem_acc) + 32 + groupID * 4 + threadID_in_group) = s_c.y;

  __syncthreads();

  int offset = 0;
  int ELTS_PER_ROW_c = N;

  offset += ROW * MMA_M * ELTS_PER_ROW_c;
  offset += COL * MMA_N;

  int TITLE_THREADS_PER_ROW_c = (MMA_N + ELTS_PER_THREADS - 1) / ELTS_PER_THREADS;
  
  
  int row_acc = tidx / TITLE_THREADS_PER_ROW_c;
  int col_acc = tidx % TITLE_THREADS_PER_ROW_c;

  offset += row_acc * ELTS_PER_ROW_c;
  offset += col_acc * ELTS_PER_THREADS;

  // __half sc_0 = *reinterpret_cast<__half*>(smem_acc + 4 * groupID *4 + 4* threadID_in_group);
  // __half sc_1 = *reinterpret_cast<__half*>(smem_acc + 4 * groupID *4 + 4* threadID_in_group + 2);
  // __half sc_2 = *reinterpret_cast<__half*>(smem_acc + 32 * 4 + 4 * groupID *4 + 4* threadID_in_group);
  // __half sc_3 = *reinterpret_cast<__half*>(smem_acc + 32 * 4 + 4 * groupID *4 + 4* threadID_in_group + 2);
  // printf("ROW: %d, COL %d, ELTS_PER_ROW_c %d THREADS_PER_ROW_c %d, row_acc %d col_acc %d offset %d sc_0 %f sc_1 %f sc_2 %f sc_3 %f\n",
  //       ROW, COL, ELTS_PER_ROW_c, TITLE_THREADS_PER_ROW_c, row_acc, col_acc, offset, __half2float(sc_0),  __half2float(sc_1),
  //       __half2float(sc_2), __half2float(sc_3));
  
  // // if (row_acc < MMA_M && col_acc < MMA_N) {
    uint4 t_a = *(reinterpret_cast<uint4*>(c + offset));
    if (row_acc < MMA_M && col_acc * ELTS_PER_THREADS < MMA_N) {
        // uint4 t_b = *(reinterpret_cast<uint4*>(smem_acc + row_acc * MMA_N * BYTES_PER_ELT + col_acc * ELTS_PER_THREADS * BYTES_PER_ELT));
        // __half sc_0 = *reinterpret_cast<__half*>(smem_acc + row_acc * MMA_N * BYTES_PER_ELT + col_acc * ELTS_PER_THREADS * BYTES_PER_ELT + 0);
        // __half sc_1 = *reinterpret_cast<__half*>(smem_acc + row_acc * MMA_N * BYTES_PER_ELT + col_acc * ELTS_PER_THREADS * BYTES_PER_ELT + 2);
        // __half sc_2 = *reinterpret_cast<__half*>(smem_acc + row_acc * MMA_N * BYTES_PER_ELT + col_acc * ELTS_PER_THREADS * BYTES_PER_ELT + 4);
        // __half sc_3 = *reinterpret_cast<__half*>(smem_acc + row_acc * MMA_N * BYTES_PER_ELT + col_acc * ELTS_PER_THREADS * BYTES_PER_ELT + 6);
        // __half sc_4 = *reinterpret_cast<__half*>(smem_acc + row_acc * MMA_N * BYTES_PER_ELT + col_acc * ELTS_PER_THREADS * BYTES_PER_ELT + 8);
        // __half sc_5 = *reinterpret_cast<__half*>(smem_acc + row_acc * MMA_N * BYTES_PER_ELT + col_acc * ELTS_PER_THREADS * BYTES_PER_ELT + 10);
        // __half sc_6 = *reinterpret_cast<__half*>(smem_acc + row_acc * MMA_N * BYTES_PER_ELT + col_acc * ELTS_PER_THREADS * BYTES_PER_ELT + 12);
        // __half sc_7 = *reinterpret_cast<__half*>(smem_acc + row_acc * MMA_N * BYTES_PER_ELT + col_acc * ELTS_PER_THREADS * BYTES_PER_ELT + 14);
        // printf("ROW: %d, COL %d, ELTS_PER_ROW_c %d THREADS_PER_ROW_c %d, row_acc %d col_acc %d offset %d sc_0 %f sc_1 %f sc_2 %f sc_3 %f sc_4 %f sc_5 %f sc_6 %f sc_7 %f\n",
        //   ROW, COL, ELTS_PER_ROW_c, TITLE_THREADS_PER_ROW_c, row_acc, col_acc, offset, __half2float(sc_0),  __half2float(sc_1),
        //    __half2float(sc_2), __half2float(sc_3), __half2float(sc_4), __half2float(sc_5), __half2float(sc_6), __half2float(sc_7));
        *(reinterpret_cast<uint4*>(c + offset)) = *(reinterpret_cast<uint4*>(smem_acc + row_acc * MMA_N * BYTES_PER_ELT + col_acc * ELTS_PER_THREADS * BYTES_PER_ELT));
    }
    // t_a = t_b;
    
    // *(reinterpret_cast<uint4*>(c + offset)) = *(reinterpret_cast<uint4*>(smem_acc + row_acc * MMA_N * BYTES_PER_ELT + col_acc * ELTS_PER_THREADS * BYTES_PER_ELT));
  // }
}


void matrix_v8_cuda(const torch::Tensor x1, const torch::Tensor x2, torch::Tensor out) {
  const auto x1_sizes = x1.sizes();
  const auto x2_sizes = x2.sizes();
  const auto x1_m = x1_sizes[0];
  const auto x1_n = x1_sizes[1];
  const auto x2_m = x2_sizes[0];
  const auto x2_n = x2_sizes[1];

  const auto out_sizes = out.sizes();
  const auto out_m = out_sizes[0];
  const auto out_n = out_sizes[1];

  AT_ASSERTM(x1_n == x2_m, "");
  AT_ASSERTM(out_m == x1_m, "");
  AT_ASSERTM(out_n == x2_n, "");

  int M = x1_m;
  int N = x2_n;
  int K = x1_n;

  // Get the raw pointers to the tensor data

  __half* x1_data = (__half*)x1.data_ptr();
  __half* x2_data = (__half*)x2.data_ptr();
  __half* out_data = (__half*)out.data_ptr();

  // Define the number of threads and blocks for the kernel launch
  int TITLE_M = (M + MMA_M - 1) / MMA_M;
  int TITLE_N = (N + MMA_N - 1) / MMA_N;
  int TITLE_K = (K + MMA_K - 1) / MMA_K;

  dim3 threads(32);
  dim3 blocks(TITLE_M, TITLE_N);

  // Launch the kernel
  time_t now = time(NULL);
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  std::cout << "sharedMemPerBlock: " << prop.sharedMemPerBlock 
            << " sharedMemPerMultiprocessor: " << prop.sharedMemPerMultiprocessor
            << " sharedMemPerBlockOptin: " << prop.sharedMemPerBlockOptin << std::endl;
  int smem_size = 1024 * 48;
  cudaFuncSetAttribute(
                MatrixForwardV8, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
  printf("threads is (%d, %d, %d); blocks is (%d, %d, %d)\n", threads.x, threads.y, threads.z, blocks.x, blocks.y, blocks.z);
  
  MatrixForwardV8<<<blocks, threads, smem_size>>>(x1_data, x2_data, out_data, M, N, K);
  // MatrixForwardV8(x1_data, x2_data, out_data, x1_m, x2_n, x1_n);
  cudaPeekAtLastError();
  time_t t = time(NULL) - now;
  printf("cuda cost is %d\n", t);
}
