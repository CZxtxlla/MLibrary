#include "../include/tensor.h"
#include "../include/autograd.h"

int main() {
    printf("--- Autograd Engine Test ---\n");

    // We will just use 1D scalars (shape [1]) for this test
    int shape[] = {1};

    // 1. Initialize our variables x and y
    Tensor* x = create_tensor(shape, 1, true);
    Tensor* y = create_tensor(shape, 1, true);

    // Set their actual data values
    x->data[0] = 2.0f;
    y->data[0] = 3.0f;

    // 2. The Forward Pass
    // a = x + y
    Tensor* a = tensor_add(x, y);
    // z = a * x
    Tensor* z = tensor_mul(a, x);

    printf("Forward Pass output (z) : %f (Expected: 10.0)\n", z->data[0]);

    // 3. The Backward Pass (Manual Trigger)
    // We always seed the final output gradient with 1.0 (since dz/dz = 1)
    z->grad[0] = 1.0f;

    // The Backward Pass (Automatic!)
    printf("Running automatic backward pass...\n");
    backward(z);

    // 4. Verify the Gradients
    printf("Gradient of x (dz/dx)   : %f (Expected: 7.0)\n", x->grad[0]);
    printf("Gradient of y (dz/dy)   : %f (Expected: 2.0)\n", y->grad[0]);

    // 5. Cleanup
    free_tensor(z);
    free_tensor(a);
    free_tensor(y);
    free_tensor(x);

    return 0;
}