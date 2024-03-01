// refer to https://github.com/Kobzol/hardware-effects-gpu/blob/master/bank-conflicts/bank-conflicts.cu
// 

#include <cuda_runtime_api.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <device_launch_parameters.h>
#include <cuda_fp16.h>
#include <stdint.h>

__global__
void bank_conflicts_cuda_v1(uint32_t* Cptr, int offset) {
    if (threadIdx.x == 0 && blockIdx.x == 0 && blockIdx.y == 0) {
        printf("bank_conflicts_cuda_v1\n");
    }
    constexpr int MEMORY_SIZE = 512;
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
    constexpr int MEMORY_SIZE = 512;
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
    constexpr int MEMORY_SIZE = 512;
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
    constexpr int MEMORY_SIZE = 512;
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
    constexpr int MEMORY_SIZE = 512;
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
        *(reinterpret_cast<uint4*>(Cptr + bidx * MEMORY_SIZE * 8 + 8 * index)) = *(reinterpret_cast<uint4*>(sharedMemV5 + 8 * index));
        // uint4 tmp = *(reinterpret_cast<uint4*>(sharedMemV5 + 8 * index));
        // printf("%d, %d, %d, %d\n", tmp.x, tmp.y, tmp.z, tmp.w);
        index += 128;
        index %= MEMORY_SIZE;
    }
}

__global__
void bank_conflicts_cuda_v6(half* Cptr, int offset) {
    if (threadIdx.x == 0 && blockIdx.x == 0 && blockIdx.y == 0) {
        printf("bank_conflicts_cuda_v6\n");
    }
    constexpr int MEMORY_SIZE = 512;
    extern __shared__ half sharedMemV6[];

    int threadId = threadIdx.x;
    int bidx = blockIdx.y;

    // registers to shared memory
    uint32_t index = threadId * offset;
    for (int i = 0; i < 10000; i++)
    {
        *(reinterpret_cast<uint4*>(sharedMemV6 + 8 * index)) = make_uint4(index * i, index * i, index * i, index * i);
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
        *(reinterpret_cast<uint4*>(Cptr + bidx * MEMORY_SIZE * 8 + 8 * index)) = *(reinterpret_cast<uint4*>(sharedMemV6 + 8 * index));
        index += 128;
        index %= MEMORY_SIZE;
    }
}

__global__
void bank_conflicts_cuda_v7(half* Cptr, int offset) {
    if (threadIdx.x == 0 && blockIdx.x == 0 && blockIdx.y == 0) {
        printf("bank_conflicts_cuda_v7\n");
    }
    // constexpr int MEMORY_SIZE = 512;
    constexpr int MEMORY_SIZE = 512;
    extern __shared__ half sharedMemV7[];

    int threadId = threadIdx.x;
    // int bidx = blockIdx.x;
    // int bidy = blockIdx.y;
    int blockId = blockIdx.x + blockIdx.y*gridDim.x + blockIdx.z*gridDim.x*gridDim.y;
    // if (threadIdx.x == 0 && blockIdx.x == 0 && blockIdx.y == 0) {
    //     printf("bank_conflicts_cuda_v7\n");
    // }
    // registers to shared memory
    uint32_t index = threadId * offset;
    for (int i = 0; i < 10000; i++)
    {
        *(reinterpret_cast<uint4*>(sharedMemV7 + 8 * index)) = make_uint4(index * i, index * i, index * i, index * i);
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
        *(reinterpret_cast<uint4*>(Cptr + blockId * MEMORY_SIZE * 8 + 8 * index)) = *(reinterpret_cast<uint4*>(sharedMemV7 + 8 * index));
        index += 128;
        index %= MEMORY_SIZE;
    }
}

__global__
void bank_conflicts_cuda_v8(half* Cptr, int offset) {
    if (threadIdx.x == 0 && blockIdx.x == 0 && blockIdx.y == 0) {
        printf("bank_conflicts_cuda_v8\n");
    }
    
    int M = 81920;
    int N = 256;
    int kTiledM = 64 * 8;
    int kTiledN = 64;
    int MEMORY_SIZE = kTiledM * kTiledN / 8;
    extern __shared__ half sharedMemV8[];

    int bidx = blockIdx.x;
    int bidy = blockIdx.y;
    int tidx = threadIdx.x;
    int tidRow = tidx / 8;
    int tidCol = tidx % 8;

    int blockIndex = blockIdx.x + blockIdx.y*gridDim.x + blockIdx.z*gridDim.x*gridDim.y;

    uint32_t index = tidx * offset;
    for (int i = 0; i < 10000; i++)
    {
        *(reinterpret_cast<uint4*>(sharedMemV8 + 8 * index)) = make_uint4(index * i, index * i, index * i, index * i);
        index += 128;
        index %= MEMORY_SIZE;
    }
    __syncthreads();

    // shared memory to global memory
    index = tidx * offset;
    for (int i = 0; i < 10000; i++)
    {

        *(reinterpret_cast<uint4*>(Cptr + blockIndex * MEMORY_SIZE * 8 + 8 * index)) = *(reinterpret_cast<uint4*>(sharedMemV8 + 8 * index));
        index += 128;
        index %= MEMORY_SIZE;
    }
}

__global__
void bank_conflicts_cuda_v9(half* Cptr, int offset) {
    if (threadIdx.x == 0 && blockIdx.x == 0 && blockIdx.y == 0) {
        printf("bank_conflicts_cuda_v9\n");
    }
    
    int M = 81920;
    int N = 256;
    int kTiledM = 64;
    int kTiledN = 64;
    int MEMORY_SIZE = kTiledM * kTiledN / 8;
    extern __shared__ half sharedMemV8[];

    int bidx = blockIdx.x;
    int bidy = blockIdx.y;
    int tidx = threadIdx.x;
    int tidRow = tidx / 8;
    int tidCol = tidx % 8;

    int blockIndex = blockIdx.x + blockIdx.y*gridDim.x + blockIdx.z*gridDim.x*gridDim.y;

    uint32_t index = tidx * offset;
    for (int i = 0; i < 10000; i++)
    {
        *(reinterpret_cast<uint4*>(sharedMemV8 + 8 * index)) = make_uint4(index * i, index * i, index * i, index * i);
        index += 128;
        index %= MEMORY_SIZE;
    }
    __syncthreads();

    // shared memory to global memory
    index = tidx * offset;
    for (int i = 0; i < 10000; i++)
    {

        *(reinterpret_cast<uint4*>(Cptr + blockIndex * MEMORY_SIZE * 8 + 8 * index)) = *(reinterpret_cast<uint4*>(sharedMemV8 + 8 * index));
        index += 128;
        index %= MEMORY_SIZE;
    }
}


__global__
void bank_conflicts_cuda_v10(half* Cptr, int offset) {
    if (threadIdx.x == 0 && blockIdx.x == 0 && blockIdx.y == 0) {
        printf("bank_conflicts_cuda_v10\n");
    }
    
    int M = 64 * 48;
    int N = 256;
    int kTiledM = 64;
    int kTiledN = 64;
    extern __shared__ half sharedMemV10[];

    int bidx = blockIdx.x;
    int bidy = blockIdx.y;
    int tidx = threadIdx.x;
    
    int tidRow = tidx / 8;
    int tidCol = tidx % 8;

    for (int i = 0; i < 4; i++)
    {
        *(reinterpret_cast<uint4*>(sharedMemV10 + i * 16 * kTiledN  + tidRow * kTiledN + tidCol * 8)) = make_uint4(0, 0, 0, 0);
    }

    __syncthreads();
    for (int j = 0; j < 100000; j++) {
    for (int i = 0; i < 4; i++)
    {
        *(reinterpret_cast<uint4*>(Cptr + bidx * kTiledM * N + bidy * kTiledN + i * 16 * N + + tidRow * N + tidCol * 8)) = *(reinterpret_cast<uint4*>(sharedMemV10 + i * 16 * kTiledN  + tidRow * kTiledN + tidCol * 8));
    }
    }
}

void bank_conflicts_v1(int offset) {
    constexpr int MEMORY_SIZE = 512;
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
    constexpr int MEMORY_SIZE = 512;
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
    constexpr int MEMORY_SIZE = 512;
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
    constexpr int MEMORY_SIZE = 512;
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
    int num_kernel = 256;
    constexpr int MEMORY_SIZE = 512;
    int sharedMemSize = MEMORY_SIZE * sizeof(uint4);
    int globalMemSize = num_kernel * MEMORY_SIZE * sizeof(uint4);
    half *Cptr;
    // char *Cptr_host = (char *)malloc(sharedMemSize);
    cudaMalloc(&Cptr, globalMemSize);
    // cudaMemcpy(Cptr, Cptr_host, sharedMemSize, cudaMemcpyHostToDevice);
    if (sharedMemSize > 48 * 1024) {
        cudaFuncSetAttribute(
                bank_conflicts_cuda_v5, cudaFuncAttributeMaxDynamicSharedMemorySize, sharedMemSize);
    }
    bank_conflicts_cuda_v5<<<num_kernel, 128, sharedMemSize>>>(Cptr, offset);
    cudaDeviceSynchronize();
}

void bank_conflicts_v6(int offset) {
    int num_kernel = 64 * 64;
    constexpr int MEMORY_SIZE = 512;
    int sharedMemSize = MEMORY_SIZE * sizeof(uint4);
    int globalMemSize = num_kernel * MEMORY_SIZE * sizeof(uint4);
    half *Cptr;
    // char *Cptr_host = (char *)malloc(sharedMemSize);
    cudaMalloc(&Cptr, globalMemSize);
    // cudaMemcpy(Cptr, Cptr_host, sharedMemSize, cudaMemcpyHostToDevice);
    cudaFuncSetAttribute(
                bank_conflicts_cuda_v6, cudaFuncAttributeMaxDynamicSharedMemorySize, sharedMemSize);
    dim3 block(1, num_kernel);
    bank_conflicts_cuda_v6<<<block, 128, sharedMemSize>>>(Cptr, offset);
    cudaDeviceSynchronize();
}

void bank_conflicts_v7(int offset) {
    int num_kernel_x = 64;
    int num_kernel_y = 64;
    // constexpr int MEMORY_SIZE = 512;
    constexpr int MEMORY_SIZE = 512;
    int sharedMemSize = MEMORY_SIZE * sizeof(uint4);
    int globalMemSize = num_kernel_x * num_kernel_y * MEMORY_SIZE * sizeof(uint4);
    half *Cptr;
    // char *Cptr_host = (char *)malloc(sharedMemSize);
    cudaMalloc(&Cptr, globalMemSize);
    // cudaMemcpy(Cptr, Cptr_host, sharedMemSize, cudaMemcpyHostToDevice);
    cudaFuncSetAttribute(
                bank_conflicts_cuda_v7, cudaFuncAttributeMaxDynamicSharedMemorySize, sharedMemSize);
    dim3 block(num_kernel_x, num_kernel_y);
    bank_conflicts_cuda_v7<<<block, 128, sharedMemSize>>>(Cptr, offset);
    cudaDeviceSynchronize();
}

void bank_conflicts_v8(int offset) {
    int M = 81920;
    int N = 256;
    int kTiledM = 64 * 8;
    int kTiledN = 64;
    int sharedMemSize = kTiledM * kTiledN * sizeof(half);
    int globalMemSize = M * N * sizeof(half);
    half *Cptr;
    // char *Cptr_host = (char *)malloc(sharedMemSize);
    cudaMalloc(&Cptr, globalMemSize);
    // cudaMemcpy(Cptr, Cptr_host, sharedMemSize, cudaMemcpyHostToDevice);
    cudaFuncSetAttribute(
                bank_conflicts_cuda_v8, cudaFuncAttributeMaxDynamicSharedMemorySize, sharedMemSize);
    dim3 block(M / kTiledM, N / kTiledN);
    dim3 thread(128,1,1);
    bank_conflicts_cuda_v8<<<block, thread, sharedMemSize>>>(Cptr, offset);
    cudaDeviceSynchronize();
}

void bank_conflicts_v9(int offset) {
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
                bank_conflicts_cuda_v9, cudaFuncAttributeMaxDynamicSharedMemorySize, sharedMemSize);
    dim3 block(M / kTiledM, N / kTiledN);
    dim3 thread(128,1,1);
    bank_conflicts_cuda_v9<<<block, thread, sharedMemSize>>>(Cptr, offset);
    cudaDeviceSynchronize();
}

void bank_conflicts_v10(int offset) {
    // int M = 81920;
    int M = 64 * 48;
    int N = 256;
    int kTiledM = 64;
    int kTiledN = 64;
    int sharedMemSize = kTiledM * kTiledN * sizeof(half);
    int globalMemSize = M * N * sizeof(half);
    half *Cptr;
    // char *Cptr_host = (char *)malloc(sharedMemSize);
    cudaMalloc(&Cptr, globalMemSize);
    // cudaMemcpy(Cptr, Cptr_host, sharedMemSize, cudaMemcpyHostToDevice);
    if (sharedMemSize > 48 * 1024) {
        cudaFuncSetAttribute(bank_conflicts_cuda_v10, cudaFuncAttributeMaxDynamicSharedMemorySize, sharedMemSize);
    }
    
    dim3 block(M / kTiledM, N / kTiledN);
    dim3 thread(128,1,1);
    bank_conflicts_cuda_v10<<<block, thread, sharedMemSize>>>(Cptr, offset);
    cudaDeviceSynchronize();
}

void bank_conflicts_deviceprop() {
    cudaDeviceProp prop;
    int count; 
    cudaGetDeviceCount(&count);
    for (int i = 0; i < count; i++) {
        cudaGetDeviceProperties(&prop, i);
        printf("device %d\n", i);
        printf("device name %s\n", prop.name);
        printf("device totalGlobalMem(MB) %d\n", prop.totalGlobalMem / 1024 / 1024);
        printf("device sharedMemPerBlock(KB) %d\n", prop.sharedMemPerBlock / 1024);
        printf("device regsPerBlock %d\n", prop.regsPerBlock);
        printf("device warpSize %d\n", prop.warpSize);
        printf("device memPitch %d\n", prop.memPitch);
        printf("device maxThreadsPerBlock %d\n", prop.maxThreadsPerBlock);
        printf("device maxThreadsDim [%d, %d, %d]\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
        printf("device maxGridSize [%d, %d, %d]\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
        printf("device clockRate %d\n", prop.clockRate);
        printf("device totalConstMem %d\n", prop.totalConstMem);
        printf("device Compute Capability %d %d\n", prop.major, prop.minor);
        printf("device textureAlignment %d\n", prop.textureAlignment);
        printf("device deviceOverlap %d\n", prop.deviceOverlap);
        printf("device multiProcessorCount %d\n", prop.multiProcessorCount);
        printf("device kernelExecTimeoutEnabled %d\n", prop.kernelExecTimeoutEnabled);
        printf("device integrated %d\n", prop.integrated);
        printf("device canMapHostMemory %d\n", prop.canMapHostMemory);
        printf("device computeMode %d\n", prop.computeMode);
        printf("device maxTexture1D %d\n", prop.maxTexture1D);
        printf("device maxTexture2D [%d, %d]\n", prop.maxTexture2D[0], prop.maxTexture2D[1]);
        printf("device maxTexture3D [%d, %d, %d]\n", prop.maxTexture3D[0], prop.maxTexture3D[1], prop.maxTexture3D[2]);
        printf("device maxTexture1DLayered [%d, %d]\n", prop.maxTexture1DLayered[0], prop.maxTexture1DLayered[1]);
        printf("device maxTexture2DLayered [%d, %d, %d]\n", prop.maxTexture2DLayered[0], prop.maxTexture2DLayered[1], prop.maxTexture2DLayered[2]);
        printf("device surfaceAlignment %d\n", prop.surfaceAlignment);
        printf("device concurrentKernels %d\n", prop.concurrentKernels);
        printf("device ECCEnabled %d\n", prop.ECCEnabled);
        printf("device pciBusID %d\n", prop.pciBusID);
        printf("device pciDeviceID %d\n", prop.pciDeviceID);
        printf("device pciDomainID %d\n", prop.pciDomainID);
        printf("device tccDriver %d\n", prop.tccDriver);
        printf("device asyncEngineCount %d\n", prop.asyncEngineCount);
        printf("device unifiedAddressing %d\n", prop.unifiedAddressing);
        printf("device memoryClockRate %d\n", prop.memoryClockRate);
        printf("device memoryBusWidth %d\n", prop.memoryBusWidth);
        printf("device l2CacheSize %d\n", prop.l2CacheSize);
        printf("device maxThreadsPerMultiProcessor %d\n", prop.maxThreadsPerMultiProcessor);
        // printf("device memPitch %d\n", prop.memPitch);
    }
}