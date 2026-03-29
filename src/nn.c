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

// MLP functions go here

MLP* create_mlp(int* architecture, int num_layers) {
    MLP* model = (MLP*)malloc(sizeof(MLP));
    if (model == NULL) {
        fprintf(stderr, "Error: Failed to allocate memory for MLP struct.\n");
        return NULL;
    }

    model->num_layers = num_layers - 1; // Number of linear layers is one less than the number of layer sizes
    model->layers = (LinearLayer**)malloc(model->num_layers * sizeof(LinearLayer*));
    if (model->layers == NULL) {
        fprintf(stderr, "Error: Failed to allocate memory for MLP layers array.\n");
        free(model);
        return NULL;
    }

    for (int i = 0; i < model->num_layers; i ++){
        model->layers[i] = create_linear_layer(architecture[i], architecture[i + 1]);
    }
    return model;
}

Tensor* mlp_forward(MLP* model, Tensor* input) {
    Tensor* current = input;
    for (int i = 0; i < model->num_layers; i++) {
        current = linear_forward(model->layers[i], current);
        if (i < model->num_layers - 1) { // Apply ReLU after all but the last layer
            current = tensor_relu(current);
        }
    }
    return current;
}

Tensor** mlp_get_parameters(MLP* model, int* out_num_paramters) {
    *out_num_paramters = model->num_layers * 2; // Each layer has a weight and a bias
    Tensor** params = (Tensor**)malloc(*out_num_paramters * sizeof(Tensor*));
    if (params == NULL) {
        fprintf(stderr, "Error: Failed to allocate memory for MLP parameters array.\n");
        return NULL;
    }
    for (int i = 0; i < model->num_layers; i++) {
        params[i * 2] = model->layers[i]->weight;
        params[i * 2 + 1] = model->layers[i]->bias;
    }
    return params;
}

void free_mlp(MLP* model) {
    if (model != NULL) {
        for (int i = 0; i < model->num_layers; i++) {
            free_linear_layer(model->layers[i]);
        }
        free(model->layers);
        free(model);
    }
}