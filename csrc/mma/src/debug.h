
// #pragma once
// #include <cuda.h>

// #define DEBUG_WARP_MATMUL_DEVICE 1

// // copy from https://stackoverflow.com/questions/20201335/add-char-arrays-in-cuda
// __device__ char * ld_mma_strcpy(char *dest, const char *src){
//   int i = 0;
//   do {
//     dest[i] = src[i];}
//   while (src[i++] != 0);
//   return dest;
// }

// __device__ char * ld_mma_strcat(char *dest, const char *src){
//   int i = 0;
//   while (dest[i] != 0) i++;
//   ld_mma_strcpy(dest+i, src);
//   return dest;
// }

// #ifdef DEBUG_WARP_MATMUL_DEVICE
// __device__ void warp_matmul_device_debug(char *func_name, char *fs, ...) 
// { 
//   printf(ld_mma_strcat("device warp matmul: %s: ", fs), func_name, ...); 
// }
// #else
// __device__ void warp_matmul_device_debug(char *func_name, char *fs, ...) {}
// #endif

// #ifdef TC397_SCU_DEBUG
// #define tc397_scu_debug(fs,...) \
//     fprintf(stderr,"tc397_scu: %s: "fs,__func__,##__VA_ARGS__)
// #else
// #define tc397_scu_debug(fs,...)
// #endif