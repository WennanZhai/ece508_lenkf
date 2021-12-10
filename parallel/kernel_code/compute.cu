#include <cstdio>
#include <cstdlib>
#include <stdio.h>
#include "helper.hpp"

extern "C"{
    #include "compute.h"
}

#define BLOCK_SIZE 512

__global__ void test_kernel(uint32_t *in, uint32_t *out, int len){
    if(threadIdx.x == 1)    printf("Kernel is running\n");
}

extern "C" 
void test_kernel_wrapper(int inputLength) {
    uint32_t *deviceInput = nullptr;
    uint32_t *deviceOutput= nullptr;

    auto hostInput = generate_input(inputLength);
    const size_t byteCount = inputLength * sizeof(uint32_t);

    timer_start("Allocating GPU memory.");
    THROW_IF_ERROR(cudaMalloc((void **)&deviceInput, byteCount));
    THROW_IF_ERROR(cudaMalloc((void **)&deviceOutput, byteCount));
    timer_stop();

    timer_start("Copying input memory to the GPU.");
    THROW_IF_ERROR(cudaMemcpy(deviceInput, hostInput.data(), byteCount,
                     cudaMemcpyHostToDevice));
    THROW_IF_ERROR(cudaMemset(deviceOutput, 0, byteCount));
    timer_stop();

    timer_start("Performing GPU Gather computation");
    int numBlocks = 8;
    test_kernel<<<numBlocks, BLOCK_SIZE>>>(deviceInput, deviceOutput, inputLength);
    timer_stop();

    std::vector<uint32_t> hostOutput(inputLength);

    timer_start("Copying output memory to the CPU");
    timer_stop();

    cudaFree(deviceInput);
    cudaFree(deviceOutput);
}