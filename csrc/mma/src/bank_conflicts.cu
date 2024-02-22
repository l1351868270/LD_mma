// refer to https://github.com/Kobzol/hardware-effects-gpu/blob/master/bank-conflicts/bank-conflicts.cu
// 

#include <cuda_runtime_api.h>
#include <stdio.h>
#include <device_launch_parameters.h>
#include <cuda_fp16.h>

__global__
void bank_conflicts_cuda(half* Cptr, int offset) {
    if (threadIdx.x == 0 && blockIdx.x == 0 && blockIdx.y == 0) {
        printf("bank_conflicts_cuda\n");
    }

    int M = 81920;
    int N = 256;
    int kTiledM = 64;
    int kTiledN = 64;

    extern __shared__ half sharedMem[];

    int bidx = blockIdx.x;
    int bidy = blockIdx.y;
    int tidx = threadIdx.x;
    
    int tidRow = tidx / 8;
    int tidCol = tidx % 8;

    for (int i = 0; i < 4; i++)
    {
        *(reinterpret_cast<uint4*>(sharedMem + i * 16 * kTiledN  + tidRow * kTiledN + tidCol * 8)) = make_uint4(0, 0, 0, 0);
    }
    __syncthreads();
    for (int i = 0; i < 4; i++)
    {
        *(reinterpret_cast<uint4*>(Cptr + bidx * kTiledM * N + bidy * kTiledN + i * 16 * N + + tidRow * N + tidCol * 8)) = *(reinterpret_cast<uint4*>(sharedMem + i * 16 * kTiledN  + tidRow * kTiledN + tidCol * 8));
    }
}

void bank_conflicts(int offset) {
    int M = 81920;
    int N = 256;
    int kTiledM = 64;
    int kTiledN = 64;
    int sharedMemSize = kTiledM * kTiledN * sizeof(half);
    int globalMemSize = M * N * sizeof(half);
    half *Cptr;
    // char *Cptr_host = (char *)malloc(sharedMemSize);
    cudaMalloc(&Cptr, globalMemSize);
    // cudaMemcpy(Cptr, Cptr_host, sharedMemSize, cudaMemcpyHostToDevice);
    cudaFuncSetAttribute(
                bank_conflicts_cuda, cudaFuncAttributeMaxDynamicSharedMemorySize, sharedMemSize);
    dim3 block(M / kTiledM, N / kTiledN);
    dim3 thread(128,1,1);
    bank_conflicts_cuda<<<block, thread, sharedMemSize>>>(Cptr, offset);
    cudaDeviceSynchronize();
}