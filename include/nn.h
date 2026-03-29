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

/// Stuff for MLP

typedef struct {
    LinearLayer** layers; // Array of pointers to the linear layers in the MLP
    int num_layers;       // Number of layers in the MLP
} MLP;

MLP* create_mlp(int* architecture, int num_layers); // Create an MLP given an array of layer sizes (e.g., [784, 128, 10]) and the number of layers, return a pointer to the MLP
Tensor* mlp_forward(MLP* model, Tensor* input); // Perform the forward pass through the MLP, return the output tensor
Tensor** mlp_get_parameters(MLP* model, int* out_num_parameters); // Get an array of pointers to all the learnable parameters (weights and biases) in the MLP, and set out_num_parameters to the total count
void free_mlp(MLP* model); // Free the memory allocated for the MLP and its layers



#endif