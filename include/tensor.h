#ifndef TENSOR_H 
#define TENSOR_H
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include "math.h"

typedef enum {
    DEVICE_CPU,
    DEVICE_GPU
} DeviceType;

typedef enum {
    OP_NONE,
    OP_ADD,
    OP_MUL,
    OP_MATMUL,
    OP_RELU,
    OP_ADDBIAS
} OpType;

typedef struct Tensor {
    // CPU memory pointers
    float* data;            // The actual forward pass values
    float* grad;            // The accumulated gradients
    //GPU memory pointers
    float* gpu_data;
    float* gpu_grad;

    DeviceType device;      // Where the tensor is currently located (CPU or GPU)

    int* shape;             // e.g., [64, 128] for a 2D matrix
    int ndims;              // Number of dimensions
    int size;               // Total number of elements (product of shape)
    
    // Autograd Graph Data
    bool requires_grad;
    struct Tensor** parents; // Pointers to the tensors that created this one
    int num_parents;
    OpType op;              // The operation that created this tensor
} Tensor;

Tensor* create_tensor(int* shape, int ndims, bool requires_grad);
void free_tensor(Tensor* t);
void tensor_to_device(Tensor* t, DeviceType device);

Tensor* tensor_add(Tensor* a, Tensor* b);
Tensor* tensor_mul(Tensor* a, Tensor* b);
Tensor* tensor_matmul(Tensor* a, Tensor* b);
Tensor* tensor_relu(Tensor* a);
Tensor* tensor_add_bias(Tensor* a, Tensor* bias);

#endif