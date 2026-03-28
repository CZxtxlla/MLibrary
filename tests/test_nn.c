#include <stdio.h>
#include "../include/tensor.h"
#include "../include/nn.h"
#include "../include/autograd.h"


int main() {
    printf("--- Neural Network (MLP) Autograd Test ---\n\n");

    // 1. Create Dummy Input Data (Shape: [1, 3] -> 1 batch, 3 features)
    int x_shape[] = {1, 3};
    Tensor* x = create_tensor(x_shape, 2, false); // Input doesn't need gradients
    x->data[0] = 1.0f; 
    x->data[1] = 2.0f; 
    x->data[2] = 3.0f;

    // 2. Instantiate the Layers
    printf("Initializing Layers...\n");
    LinearLayer* layer1 = create_linear_layer(3, 2);
    LinearLayer* layer2 = create_linear_layer(2, 1);

    // 3. The Forward Pass
    printf("Running Forward Pass...\n");
    Tensor* z1 = linear_forward(layer1, x);
    Tensor* a1 = tensor_relu(z1);
    Tensor* final_out = linear_forward(layer2, a1);

    printf("\nFinal Output Value: %f\n", final_out->data[0]);
    // Note: Because we initialized weights randomly, your output value 
    // will be slightly different every time you run this!

    // 4. The Backward Pass
    printf("\nRunning Backward Pass...\n");
    // Seed the final gradient (assuming this is our loss)
    final_out->grad[0] = 1.0f; 
    backward(final_out);

    // 5. Verify Gradients Flowed All the Way Back
    // If the chain rule worked through the matmuls, adds, and relus,
    // the very first layer should now have non-zero gradients.
    printf("\nLayer 1 Weight Gradients (should be non-zero):\n");
    for (int i = 0; i < layer1->weight->size; i++) {
        printf("dW1[%d] = %f\n", i, layer1->weight->grad[i]);
    }

    printf("\nLayer 1 Bias Gradients (should be non-zero):\n");
    for (int i = 0; i < layer1->bias->size; i++) {
        printf("db1[%d] = %f\n", i, layer1->bias->grad[i]);
    }

    free_graph(final_out); // Free the entire computation graph starting from the output

    // 6. Memory Cleanup (Freeing a graph backwards is tricky, but we 
    // free the root nodes and layer parameters here to keep it mostly clean)
    free_linear_layer(layer1);
    free_linear_layer(layer2);
    free_tensor(x);
    // Note: In a production C engine, we would write a graph-freeing utility 
    // to clean up z1, a1, and final_out, but we'll let the OS reclaim it for this test.

    printf("\nTest Complete!\n");
    return 0;
}