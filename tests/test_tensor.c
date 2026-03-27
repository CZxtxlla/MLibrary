#include <stdio.h>
#include "../include/tensor.h"

int main() {
    printf("Testing Tensor Allocation...\n");

    // Create a 2D shape: [3, 4]
    int shape[] = {3, 4};
    
    // Create the tensor (ndims=2, requires_grad=true)
    Tensor* t = create_tensor(shape, 2, true);

    if (t != NULL) {
        printf("Success! Allocated a tensor of size %d\n", t->size);
        printf("Shape: [%d, %d]\n", t->shape[0], t->shape[1]);
        
        // Free the memory
        free_tensor(t);
        printf("Memory freed successfully.\n");
    } else {
        printf("Failed to allocate tensor.\n");
    }

    return 0;
}