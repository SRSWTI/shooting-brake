#include <cstdint>
#include <cstdio>
#include <cuda_runtime.h>

extern "C" {

__global__ void sb_write_flag_kernel(unsigned int *flag, unsigned int value) {
    *flag = value;
    __threadfence_system();
}

__global__ void sb_wait_flag_kernel(volatile unsigned int *flag,
                                     unsigned int target) {
    while (*flag != target) {
        __threadfence_system();
    }
}

void sb_write_flag(int64_t flag_ptr, unsigned int value) {
    sb_write_flag_kernel<<<1, 1>>>((unsigned int *)flag_ptr, value);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "sb_write_flag launch error: %s (ptr=%ld val=%u)\n",
                cudaGetErrorString(err), (long)flag_ptr, value);
    }
}

void sb_wait_flag(int64_t flag_ptr, unsigned int target) {
    sb_wait_flag_kernel<<<1, 1>>>((volatile unsigned int *)flag_ptr, target);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "sb_wait_flag launch error: %s (ptr=%ld target=%u)\n",
                cudaGetErrorString(err), (long)flag_ptr, target);
    }
}

}
