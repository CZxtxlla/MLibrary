#include <stdio.h>
#include "../include/autograd.h"

int main() {
    printf("--- Matrix Multiplication Autograd Test ---\n\n");

    int shape[] = {2, 2};

    // 1. Create Tensors A and B
    Tensor* a = create_tensor(shape, 2, true);
    Tensor* b = create_tensor(shape, 2, true);

    // Populate A: [1, 2]
    //             [3, 4]
    a->data[0] = 1.0f; a->data[1] = 2.0f;
    a->data[2] = 3.0f; a->data[3] = 4.0f;

    // Populate B: [5, 6]
    //             [7, 8]
    b->data[0] = 5.0f; b->data[1] = 6.0f;
    b->data[2] = 7.0f; b->data[3] = 8.0f;

    // 2. The Forward Pass
    Tensor* c = tensor_matmul(a, b);

    printf("Forward Pass (C = A @ B):\n");
    printf("[ %f, %f ]  (Expected: 19.0, 22.0)\n", c->data[0], c->data[1]);
    printf("[ %f, %f ]  (Expected: 43.0, 50.0)\n\n", c->data[2], c->data[3]);

    // 3. The Backward Pass Setup
    // Our backward() function assumes a scalar root and only sets grad[0] = 1.0.
    // For this matrix test, we manually set ALL gradients of C to 1.0 to simulate 
    // an evenly distributed loss, then call backward().
    for (int i = 0; i < c->size; i++) {
        c->grad[i] = 1.0f;
    }
    
    printf("Running automatic backward pass...\n\n");
    backward(c); // This will overwrite c->grad[0] with 1.0, which is perfectly fine.

    // 4. Verify the Gradients
    printf("Gradient of A (dA):\n");
    printf("[ %f, %f ]  (Expected: 11.0, 15.0)\n", a->grad[0], a->grad[1]);
    printf("[ %f, %f ]  (Expected: 11.0, 15.0)\n\n", a->grad[2], a->grad[3]);

    printf("Gradient of B (dB):\n");
    printf("[ %f,  %f ]  (Expected: 4.0, 4.0)\n", b->grad[0], b->grad[1]);
    printf("[ %f,  %f ]  (Expected: 6.0, 6.0)\n", b->grad[2], b->grad[3]);

    // 5. Cleanup
    free_tensor(c);
    free_tensor(b);
    free_tensor(a);

    return 0;
}