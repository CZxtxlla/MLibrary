#include "../include/nn.h"

float random_float() {
    // Helper function to generate a random float between -1.0 and 1.0 for weight initialization
    return ((float)rand() / (float)RAND_MAX) * 2.0f - 1.0f;
}

LinearLayer* create_linear_layer(int in_features, int out_features) {
    // Create a linear layer with randomly initialized weights and zero-initialized bias, return a pointer to the layer
    LinearLayer* layer = (LinearLayer*)malloc(sizeof(LinearLayer));
    if (layer == NULL) {
        fprintf(stderr, "Error: Failed to allocate memory for LinearLayer struct.\n");
        return NULL;
    }

    int weight_shape[] = {in_features, out_features}; // Shape of the weight tensor
    layer->weight = create_tensor(weight_shape, 2, true);
    if (layer->weight == NULL) {
        free(layer);
        return NULL;
    }

    int bias_shape[] = {1, out_features}; // Shape of the bias tensor
    layer->bias = create_tensor(bias_shape, 2, true);
    if (layer->bias == NULL) {
        free_tensor(layer->weight);
        free(layer);
        return NULL;
    }

    // Initialize weights randomly and bias to zero
    for (int i = 0; i < layer->weight->size; i++) {
        layer->weight->data[i] = random_float(); // Small random values between -1.0 and 1.0
    }
    for (int i = 0; i < layer->bias->size; i++) {
        layer->bias->data[i] = 0.0f; // Bias initialized to zero
    }
    return layer;
}

Tensor* linear_forward(LinearLayer* layer, Tensor* input) {
    // Perform the forward pass through the linear layer and return the output tensor
    Tensor* output = tensor_matmul(input, layer->weight); // output = input @ weight

    Tensor* result = tensor_add_bias(output, layer->bias); // result = output + bias

    // Keep the intermediate alive: autograd stores it as a parent of result.
    // Freeing it here causes use-after-free during backward().
    return result;
}

void free_linear_layer(LinearLayer* layer) {
    // Free the memory allocated for the linear layer and its tensors
    if (layer != NULL) {
        free_tensor(layer->weight);
        free_tensor(layer->bias);
        free(layer);
    }
}