// refer to https://github.com/Kobzol/hardware-effects-gpu/blob/master/bank-conflicts/bank-conflicts.cu
// 

#include <cuda_runtime_api.h>
#include <stdio.h>
#include <device_launch_parameters.h>
#include <cuda_fp16.h>

__global__
void bank_conflicts_cuda_v1(uint32_t* Cptr, int offset) {
    if (threadIdx.x == 0 && blockIdx.x == 0 && blockIdx.y == 0) {
        printf("bank_conflicts_cuda_v1\n");
    }
    constexpr int MEMORY_SIZE = 4096;
    __shared__ uint32_t sharedMem[MEMORY_SIZE];

    int threadId = threadIdx.x;

    // init shared memory
    if (threadId == 0)
    {
        for (int i = 0; i < MEMORY_SIZE; i++) sharedMem[i] = 0;
    }
    __syncthreads();

    // registers to shared memory
    uint32_t index = threadId * offset;
    for (int i = 0; i < 10000; i++)
    {
        sharedMem[index] += index * i;
        index += 128;
        index %= MEMORY_SIZE;
    }
    __syncthreads();

    // shared memory to global memory
    index = threadId * offset;
    for (int i = 0; i < 4; i++)
    {
        *(Cptr + index) = sharedMem[index];
        index += 128;
        index %= MEMORY_SIZE;
    }

}

__global__
void bank_conflicts_cuda_v2(uint64_t* Cptr, int offset) {
    if (threadIdx.x == 0 && blockIdx.x == 0 && blockIdx.y == 0) {
        printf("bank_conflicts_cuda_v2\n");
    }
    constexpr int MEMORY_SIZE = 4096;
    __shared__ uint64_t sharedMem[MEMORY_SIZE];

    int threadId = threadIdx.x;

    // init shared memory
    if (threadId == 0)
    {
        for (int i = 0; i < MEMORY_SIZE; i++) sharedMem[i] = 0;
    }
    __syncthreads();

    // registers to shared memory
    uint32_t index = threadId * offset;
    for (int i = 0; i < 10000; i++)
    {
        sharedMem[index] += index * i;
        index += 128;
        index %= MEMORY_SIZE;
    }
    __syncthreads();

    // shared memory to global memory
    index = threadId * offset;
    for (int i = 0; i < 10000; i++)
    {
        *(Cptr + index) = sharedMem[index];
        index += 128;
        index %= MEMORY_SIZE;
    }
}

__global__
void bank_conflicts_cuda_v3(uint4* Cptr, int offset) {
    if (threadIdx.x == 0 && blockIdx.x == 0 && blockIdx.y == 0) {
        printf("bank_conflicts_cuda_v3\n");
    }
    constexpr int MEMORY_SIZE = 4096;
    extern __shared__ uint4 sharedMem[];

    int threadId = threadIdx.x;

    // init shared memory
    if (threadId == 0)
    {
        for (int i = 0; i < MEMORY_SIZE; i++) sharedMem[i] = make_uint4(0, 0, 0, 0);
    }
    __syncthreads();

    // registers to shared memory
    uint32_t index = threadId * offset;
    for (int i = 0; i < 10000; i++)
    {
        sharedMem[index] = make_uint4(index * i, index * i, index * i, index * i);
        index += 128;
        index %= MEMORY_SIZE;
    }
    __syncthreads();

    // shared memory to global memory
    index = threadId * offset;
    for (int i = 0; i < 10000; i++)
    {
        // (*(Cptr + index)).x = sharedMem[index].x;
        // (*(Cptr + index)).y = sharedMem[index].y;
        // (*(Cptr + index)).z = sharedMem[index].z;
        // (*(Cptr + index)).w = sharedMem[index].w;
        *(Cptr + index) = sharedMem[index];
        index += 128;
        index %= MEMORY_SIZE;
    }
}

__global__
void bank_conflicts_cuda_v4(half* Cptr, int offset) {
    if (threadIdx.x == 0 && blockIdx.x == 0 && blockIdx.y == 0) {
        printf("bank_conflicts_cuda_v4\n");
    }
    constexpr int MEMORY_SIZE = 4096;
    extern __shared__ half sharedMemV4[];

    int threadId = threadIdx.x;

    // init shared memory
    if (threadId == 0)
    {
        for (int i = 0; i < MEMORY_SIZE; i++) *(reinterpret_cast<uint4*>(sharedMemV4 + 8 * i)) = make_uint4(0, 0, 0, 0);
    }
    __syncthreads();

    // registers to shared memory
    uint32_t index = threadId * offset;
    for (int i = 0; i < 10000; i++)
    {
        *(reinterpret_cast<uint4*>(sharedMemV4 + 8 * index)) = make_uint4(index * i, index * i, index * i, index * i);
        index += 128;
        index %= MEMORY_SIZE;
    }
    __syncthreads();

    // shared memory to global memory
    index = threadId * offset;
    for (int i = 0; i < 10000; i++)
    {
        // (*(Cptr + index)).x = sharedMem[index].x;
        // (*(Cptr + index)).y = sharedMem[index].y;
        // (*(Cptr + index)).z = sharedMem[index].z;
        // (*(Cptr + index)).w = sharedMem[index].w;
        *(reinterpret_cast<uint4*>(Cptr + 8 * index)) = *(reinterpret_cast<uint4*>(sharedMemV4 + 8 * index));
        index += 128;
        index %= MEMORY_SIZE;
    }
}

__global__
void bank_conflicts_cuda_v5(half* Cptr, int offset) {
    if (threadIdx.x == 0 && blockIdx.x == 0 && blockIdx.y == 0) {
        printf("bank_conflicts_cuda_v5\n");
    }
    constexpr int MEMORY_SIZE = 4096;
    extern __shared__ half sharedMemV5[];

    int threadId = threadIdx.x;
    int bidx = blockIdx.x;

    // registers to shared memory
    uint32_t index = threadId * offset;
    for (int i = 0; i < 10000; i++)
    {
        *(reinterpret_cast<uint4*>(sharedMemV5 + 8 * index)) = make_uint4(index * i, index * i, index * i, index * i);
        index += 128;
        index %= MEMORY_SIZE;
    }
    __syncthreads();

    // shared memory to global memory
    index = threadId * offset;
    for (int i = 0; i < 10000; i++)
    {
        // (*(Cptr + index)).x = sharedMem[index].x;
        // (*(Cptr + index)).y = sharedMem[index].y;
        // (*(Cptr + index)).z = sharedMem[index].z;
        // (*(Cptr + index)).w = sharedMem[index].w;
        *(reinterpret_cast<uint4*>(Cptr + bidx * MEMORY_SIZE * 8 + 8 * index)) = *(reinterpret_cast<uint4*>(sharedMemV5 + 8 * index));
        index += 128;
        index %= MEMORY_SIZE;
    }
}

__global__
void bank_conflicts_cuda_v6(half* Cptr, int offset) {
    if (threadIdx.x == 0 && blockIdx.x == 0 && blockIdx.y == 0) {
        printf("bank_conflicts_cuda_v6\n");
    }
    
    int M = 81920;
    int N = 256;
    int kTiledM = 64;
    int kTiledN = 64;
    int MEMORY_SIZE = kTiledM * kTiledM / 8;
    extern __shared__ half sharedMemV6[];

    int bidx = blockIdx.x;
    int bidy = blockIdx.y;
    int tidx = threadIdx.x;
    int tidRow = tidx / 8;
    int tidCol = tidx % 8;

    int blockIndex = blockIdx.x + blockIdx.y*gridDim.x + blockIdx.z*gridDim.x*gridDim.y;

    uint32_t index = tidx * offset;
    for (int i = 0; i < 10000; i++)
    {
        *(reinterpret_cast<uint4*>(sharedMemV6 + 8 * index)) = make_uint4(index * i, index * i, index * i, index * i);
        index += 128;
        index %= MEMORY_SIZE;
    }
    __syncthreads();

    // shared memory to global memory
    index = tidx * offset;
    for (int i = 0; i < 10000; i++)
    {

        *(reinterpret_cast<uint4*>(Cptr + blockIndex * MEMORY_SIZE * 8 + 8 * index)) = *(reinterpret_cast<uint4*>(sharedMemV6 + 8 * index));
        index += 128;
        index %= MEMORY_SIZE;
    }
}

// __global__
// void bank_conflicts_cuda_v5(half* Cptr, int offset) {
//     if (threadIdx.x == 0 && blockIdx.x == 0 && blockIdx.y == 0) {
//         printf("bank_conflicts_cuda_v5\n");
//     }
    
//     int M = 81920;
//     int N = 256;
//     int kTiledM = 64;
//     int kTiledN = 64;
//     int MEMORY_SIZE = kTiledM * kTiledN / 8;
//     extern __shared__ half sharedMemV5[];

//     int bidx = blockIdx.x;
//     int bidy = blockIdx.y;
//     int tidx = threadIdx.x;
    
//     int tidRow = tidx / 8;
//     int tidCol = tidx % 8;

//     // for (int i = 0; i < 4; i++)
//     // {
//     //     *(reinterpret_cast<uint4*>(sharedMemV5 + i * 16 * kTiledN  + tidRow * kTiledN + tidCol * 8)) = make_uint4(0, 0, 0, 0);
//     // }

//     // __syncthreads();
//     // for (int i = 0; i < 4; i++)
//     // {
//     //     *(reinterpret_cast<uint4*>(Cptr + bidx * kTiledM * N + bidy * kTiledN + i * 16 * N + + tidRow * N + tidCol * 8)) = *(reinterpret_cast<uint4*>(sharedMemV5 + i * 16 * kTiledN  + tidRow * kTiledN + tidCol * 8));
//     // }

//     // for (int i = 0; i < MEMORY_SIZE; i++)
//     // {
//     //     *(reinterpret_cast<uint4*>(sharedMemV5 + 8 * i)) = make_uint4(0, 0, 0, 0);
//     // }
//     uint32_t index = tidx * offset;
//     for (int i = 0; i < 10000; i++)
//     {
//         // (*(Cptr + index)).x = sharedMem[index].x;
//         // (*(Cptr + index)).y = sharedMem[index].y;
//         // (*(Cptr + index)).z = sharedMem[index].z;
//         // (*(Cptr + index)).w = sharedMem[index].w;
//         *(reinterpret_cast<uint4*>(Cptr + 8 * index)) = *(reinterpret_cast<uint4*>(sharedMemV5 + 8 * index));
//         index += 128;
//         index %= MEMORY_SIZE;
//     }
// }

void bank_conflicts_v1(int offset) {
    constexpr int MEMORY_SIZE = 4096;
    int sharedMemSize = MEMORY_SIZE * sizeof(uint32_t);
    int globalMemSize = MEMORY_SIZE * sizeof(uint32_t);
    uint32_t *Cptr;
    // char *Cptr_host = (char *)malloc(sharedMemSize);
    cudaMalloc(&Cptr, globalMemSize);
    // cudaMemcpy(Cptr, Cptr_host, sharedMemSize, cudaMemcpyHostToDevice);
    cudaFuncSetAttribute(
                bank_conflicts_cuda_v1, cudaFuncAttributeMaxDynamicSharedMemorySize, sharedMemSize);

    bank_conflicts_cuda_v1<<<1, 128, sharedMemSize>>>(Cptr, offset);
    cudaDeviceSynchronize();
}

void bank_conflicts_v2(int offset) {
    constexpr int MEMORY_SIZE = 4096;
    int sharedMemSize = MEMORY_SIZE * sizeof(uint64_t);
    int globalMemSize = MEMORY_SIZE * sizeof(uint64_t);
    uint64_t *Cptr;
    // char *Cptr_host = (char *)malloc(sharedMemSize);
    cudaMalloc(&Cptr, globalMemSize);
    // cudaMemcpy(Cptr, Cptr_host, sharedMemSize, cudaMemcpyHostToDevice);
    cudaFuncSetAttribute(
                bank_conflicts_cuda_v2, cudaFuncAttributeMaxDynamicSharedMemorySize, sharedMemSize);

    bank_conflicts_cuda_v2<<<1, 128, sharedMemSize>>>(Cptr, offset);
    cudaDeviceSynchronize();
}

void bank_conflicts_v3(int offset) {
    constexpr int MEMORY_SIZE = 4096;
    int sharedMemSize = MEMORY_SIZE * sizeof(uint4);
    int globalMemSize = MEMORY_SIZE * sizeof(uint4);
    uint4 *Cptr;
    // char *Cptr_host = (char *)malloc(sharedMemSize);
    cudaMalloc(&Cptr, globalMemSize);
    // cudaMemcpy(Cptr, Cptr_host, sharedMemSize, cudaMemcpyHostToDevice);
    cudaFuncSetAttribute(
                bank_conflicts_cuda_v3, cudaFuncAttributeMaxDynamicSharedMemorySize, sharedMemSize);

    bank_conflicts_cuda_v3<<<1, 128, sharedMemSize>>>(Cptr, offset);
    cudaDeviceSynchronize();
}

void bank_conflicts_v4(int offset) {
    constexpr int MEMORY_SIZE = 4096;
    int sharedMemSize = MEMORY_SIZE * sizeof(uint4);
    int globalMemSize = MEMORY_SIZE * sizeof(uint4);
    half *Cptr;
    // char *Cptr_host = (char *)malloc(sharedMemSize);
    cudaMalloc(&Cptr, globalMemSize);
    // cudaMemcpy(Cptr, Cptr_host, sharedMemSize, cudaMemcpyHostToDevice);
    cudaFuncSetAttribute(
                bank_conflicts_cuda_v4, cudaFuncAttributeMaxDynamicSharedMemorySize, sharedMemSize);

    bank_conflicts_cuda_v4<<<1, 128, sharedMemSize>>>(Cptr, offset);
    cudaDeviceSynchronize();
}

void bank_conflicts_v5(int offset) {
    int num_kernel = 20;
    constexpr int MEMORY_SIZE = 4096;
    int sharedMemSize = MEMORY_SIZE * sizeof(uint4);
    int globalMemSize = 20 * MEMORY_SIZE * sizeof(uint4);
    half *Cptr;
    // char *Cptr_host = (char *)malloc(sharedMemSize);
    cudaMalloc(&Cptr, globalMemSize);
    // cudaMemcpy(Cptr, Cptr_host, sharedMemSize, cudaMemcpyHostToDevice);
    cudaFuncSetAttribute(
                bank_conflicts_cuda_v5, cudaFuncAttributeMaxDynamicSharedMemorySize, sharedMemSize);

    bank_conflicts_cuda_v5<<<num_kernel, 128, sharedMemSize>>>(Cptr, offset);
    cudaDeviceSynchronize();
}

void bank_conflicts_v6(int offset) {
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
                bank_conflicts_cuda_v6, cudaFuncAttributeMaxDynamicSharedMemorySize, sharedMemSize);
    dim3 block(M / kTiledM, N / kTiledN);
    dim3 thread(128,1,1);
    bank_conflicts_cuda_v6<<<block, thread, sharedMemSize>>>(Cptr, offset);
    cudaDeviceSynchronize();
}

// void bank_conflicts_v5(int offset) {
//     int M = 81920;
//     int N = 256;
//     int kTiledM = 64;
//     int kTiledN = 64;
//     int sharedMemSize = kTiledM * kTiledN * sizeof(half);
//     int globalMemSize = M * N * sizeof(half);
//     half *Cptr;
//     // char *Cptr_host = (char *)malloc(sharedMemSize);
//     cudaMalloc(&Cptr, globalMemSize);
//     // cudaMemcpy(Cptr, Cptr_host, sharedMemSize, cudaMemcpyHostToDevice);
//     cudaFuncSetAttribute(
//                 bank_conflicts_cuda_v5, cudaFuncAttributeMaxDynamicSharedMemorySize, sharedMemSize);
//     dim3 block(M / kTiledM, N / kTiledN);
//     dim3 thread(128,1,1);
//     bank_conflicts_cuda_v5<<<block, thread, sharedMemSize>>>(Cptr, offset);
//     cudaDeviceSynchronize();
// }