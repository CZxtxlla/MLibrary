#include "../include/cuda_backend.h"
#include <cuda_runtime.h>

// This file will contain CUDA-specific implementations of tensor operations.

// Utility macro for checking CUDA errors
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA Error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(1); \
        } \
    } while(0)

extern "C" void cuda_allocate_device_memory(Tensor *t) {
    // Allocate GPU memory for both data and gradient if not already allocated
    size_t bytes = t->size * sizeof(float);
    // Only allocate if we haven't already!
    if (t->gpu_data == NULL) {
        CUDA_CHECK(cudaMalloc((void**)&t->gpu_data, bytes));
    }
    if (t->gpu_grad == NULL) {
        CUDA_CHECK(cudaMalloc((void**)&t->gpu_grad, bytes));
    }
}

extern "C" void cuda_free_device_memory(Tensor* t) {
    // Free GPU memory for both data and gradient if allocated
    if (t->gpu_data != NULL) {
        CUDA_CHECK(cudaFree(t->gpu_data));
        t->gpu_data = NULL;
    }
    if (t->gpu_grad != NULL) {
        CUDA_CHECK(cudaFree(t->gpu_grad));
        t->gpu_grad = NULL;
    }
}

extern "C" void cuda_copy_to_device(Tensor* t) {
    size_t bytes = t->size * sizeof(float);
    cuda_allocate_device_memory(t); // Ensure memory is allocated before copying
    
    // Copy data and gradient from host to device
    CUDA_CHECK(cudaMemcpy(t->gpu_data, t->data, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(t->gpu_grad, t->grad, bytes, cudaMemcpyHostToDevice));

}

extern "C" void cuda_copy_to_host(Tensor* t) {
    if (t -> gpu_data == NULL) { 
        fprintf(stderr, "Error: No GPU data to copy from for tensor of size %d\n", t->size);
        return;
    }
    size_t bytes = t->size * sizeof(float);
    CUDA_CHECK(cudaMemcpy(t->data, t->gpu_data, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(t->grad, t->gpu_grad, bytes, cudaMemcpyDeviceToHost));
}

