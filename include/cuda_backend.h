#ifndef CUDA_BACKEND_H
#define CUDA_BACKEND_H

#include "tensor.h"

#ifdef __cplusplus
extern "C" {
#endif

// Allocates VRAM on the GPU
void cuda_allocate_device_memory(Tensor* t);

// Frees VRAM on the GPU
void cuda_free_device_memory(Tensor* t);

// Copies float arrays from CPU RAM -> GPU VRAM
void cuda_copy_to_device(Tensor* t);

// Copies float arrays from GPU VRAM -> CPU RAM
void cuda_copy_to_host(Tensor* t);

#ifdef __cplusplus
}
#endif

#endif 