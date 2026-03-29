#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "../include/tensor.h"
#include "../include/nn.h"
#include "../include/autograd.h"
#include "../include/optim.h"

int main() {
    printf("--- MLibrary XOR Training --- \n\n");

    // 1. Setup the XOR Data (Batch Size: 4, Features: 2)
    int x_shape[] = {4, 2};
    Tensor* x = create_tensor(x_shape, 2, false);
    
    // Row 0: [0, 0]
    x->data[0] = 0.0f; x->data[1] = 0.0f;
    // Row 1: [0, 1]
    x->data[2] = 0.0f; x->data[3] = 1.0f;
    // Row 2: [1, 0]
    x->data[4] = 1.0f; x->data[5] = 0.0f;
    // Row 3: [1, 1]
    x->data[6] = 1.0f; x->data[7] = 1.0f;

    // The target answers for those 4 rows
    float targets[] = {0.0f, 1.0f, 1.0f, 0.0f};

    // 2. Setup the Network (2 inputs -> 8 hidden -> 1 output)
    LinearLayer* layer1 = create_linear_layer(2, 8);
    LinearLayer* layer2 = create_linear_layer(8, 1);

    // 3. Setup Optimizer (Higher learning rate for XOR)
    Tensor* parameters[] = {layer1->weight, layer1->bias, layer2->weight, layer2->bias};
    SGD* optim = sgd_create(parameters, 4, 0.1f); 

    printf("Starting Training...\n\n");

    // 4. The Training Loop
    int epochs = 500;
    for (int epoch = 1; epoch <= epochs; epoch++) {
        
        sgd_zero_grad(optim);

        // Forward Pass (Processing all 4 rows simultaneously!)
        Tensor* z1 = linear_forward(layer1, x);
        Tensor* a1 = tensor_relu(z1);
        Tensor* final_out = linear_forward(layer2, a1);

        // Calculate MSE Loss for the entire batch
        float total_loss = 0.0f;
        for (int i = 0; i < 4; i++) {
            float diff = final_out->data[i] - targets[i];
            total_loss += diff * diff;
            
            // Seed the gradient for each of the 4 batch outputs
            // Derivative of MSE is 2 * (pred - target), averaged over the batch
            final_out->grad[i] = 2.0f * diff / 4.0f; 
        }
        float mse = total_loss / 4.0f;

        // Backward Pass & Optimize
        backward(final_out);
        sgd_step(optim);

        // Memory Cleanup
        free_graph(final_out);

        if (epoch % 10 == 0 || epoch == 1) {
            printf("Epoch %4d | Loss: %8.4f\n", epoch, mse);
        }
    }

    printf("\nTraining Complete! Let's test the final predictions:\n\n");
    
    // 5. Final Evaluation Pass
    Tensor* z1 = linear_forward(layer1, x);
    Tensor* a1 = tensor_relu(z1);
    Tensor* final_out = linear_forward(layer2, a1);
    
    for (int i = 0; i < 4; i++) {
        printf("Input: [%.0f, %.0f] | Target: %.0f | Prediction: %8.4f\n", 
               x->data[i*2], x->data[i*2+1], targets[i], final_out->data[i]);
    }
    
    // Final memory sweep
    free_graph(final_out);
    free_linear_layer(layer1);
    free_linear_layer(layer2);
    free_tensor(x);
    sgd_free(optim);

    return 0;
}