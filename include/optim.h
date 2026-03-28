#ifndef OPTIM_H
#define OPTIM_H

#include "tensor.h"

// The SGD Optimizer struct
typedef struct {
    Tensor** parameters;    // Array of pointers to the tensors we want to update (weights/biases)
    int num_parameters;     // How many tensors are in that array
    float lr;               // The learning rate (alpha)
} SGD;

// Allocates the optimizer and stores the pointers to the learnable parameters
SGD* sgd_create(Tensor** parameters, int num_parameters, float lr);

// Loops through all parameters and applies the learning rule: data = data - (lr * grad)
void sgd_step(SGD* optim);

// Resets all parameter gradients to 0 before the next forward pass
void sgd_zero_grad(SGD* optim);

// Frees the optimizer struct itself (but not the parameters it points to)
void sgd_free(SGD* optim);

#endif