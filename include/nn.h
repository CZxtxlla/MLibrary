#ifndef NN_H
#define NN_H

#include "tensor.h"

typedef struct {
    Tensor* weight; // Shape: [in_features, out_features]
    Tensor* bias;   // Shape: [1, out_features]
} LinearLayer;

LinearLayer* create_linear_layer(int in_features, int out_features); // Create a linear layer with randomly initialized weights and zero-initialized bias, return a pointer to the layer
Tensor* linear_forward(LinearLayer* layer, Tensor* input); // Perform the forward pass through the linear layer, return the output tensor
void free_linear_layer(LinearLayer* layer); // Free the memory allocated for the linear layer and its tensors

#endif