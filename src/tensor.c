#include "../include/tensor.h"


Tensor* create_tensor(int*shape, int ndims, bool requires_grad) {
    Tensor* t = (Tensor*)malloc(sizeof(Tensor));
    if (t == NULL) {
        fprintf(stderr, "Error: Failed to allocate memory for Tensor struct.\n");
        return NULL;    
    }
    t -> ndims = ndims;
    t -> requires_grad = requires_grad;
    t -> op = OP_NONE;
    t -> parents = NULL;
    t -> num_parents = 0;

    t -> size = 1;
    t -> shape = (int*)malloc(ndims * sizeof(int)); // Allocate memory for shape array
    if (t -> shape == NULL) {
        fprintf(stderr, "Error: Failed to allocate memory for Tensor shape.\n");
        free(t); // Free the previously allocated Tensor struct
        return NULL;
    }

    for (int i = 0; i < ndims; i ++) {
        t -> shape[i] = shape[i];
        t -> size *= shape[i];
    }

    t -> data = (float*)calloc(t -> size, sizeof(float)); // Allocate and zero-initialize data array

    if(requires_grad) {
        t -> grad = (float*)calloc(t -> size, sizeof(float)); // Allocate and zero-intiialize gradient array
        if (t -> grad == NULL) {
            fprintf(stderr, "Error: Failed to allocate memory for Tensor gradient.\n");
            free(t -> shape); // Free the previously allocated shape array
            free(t); // Free the previously allocated Tensor struct
            return NULL;
        }
    } else {
        t -> grad = NULL; // No gradient needed, set to NULL
    }

    return t;
}

void free_tensor(Tensor* t) {
    if (t == NULL) return; // Nothing to free

    if (t->data != NULL) free(t->data);
    if (t->grad != NULL) free(t->grad);
    if (t->shape != NULL) free(t->shape);
    if (t->parents != NULL) free(t->parents); // Free the parents array if it exists, but not the tensors it points to

    free(t); // Free the tensor struct itself
}

Tensor* tensor_add(Tensor* a, Tensor *b) {
    //Add two tensors element-wise, return a new tensor
    if (a -> ndims != b -> ndims) {
        fprintf(stderr, "Error: Tensors must have the same number of dimensions for addition.\n");
        return NULL;
    }
    for (int i = 0; i < a -> ndims; i ++) {
        if (a -> shape[i] != b -> shape[i]) {
            fprintf(stderr, "Error: Tensors must have the same shape for addition.\n");
            return NULL;
        }
    }
    Tensor* result = create_tensor(a -> shape, a -> ndims, a -> requires_grad || b -> requires_grad);
    if (result == NULL) {
        fprintf(stderr, "Error: Failed to create result tensor for addition.\n");
        return NULL;
    }
    for (int i = 0; i < a -> size; i ++) {
        result -> data[i] = a -> data[i] + b -> data[i];
    }
    if (result -> requires_grad) {
        result -> parents = (Tensor**)malloc(2 * sizeof(Tensor*)); 
        if (result -> parents == NULL) {
            fprintf(stderr, "Error: Failed to allocate memory for parents array in addition result tensor.\n");
            free_tensor(result);
            return NULL;
        }
        // Set the parents and operation type for autograd
        result -> parents[0] = a;
        result -> parents[1] = b;
        result -> num_parents = 2;
        result -> op = OP_ADD;
    }
    return result;
}

Tensor* tensor_mul(Tensor* a, Tensor *b) {
    //Multiply two tensors element-wise, return a new tensor
    if (a -> ndims != b -> ndims) {
        fprintf(stderr, "Error: Tensors must have the same number of dimensions for multiplication.\n");
        return NULL;
    }
    for (int i = 0; i < a -> ndims; i ++) {
        if (a -> shape[i] != b -> shape[i]) {
            fprintf(stderr, "Error: Tensors must have the same shape for multiplication.\n");
            return NULL;
        }
    }
    Tensor* result = create_tensor(a -> shape, a -> ndims, a -> requires_grad || b -> requires_grad);
    if (result == NULL) {
        fprintf(stderr, "Error: Failed to create result tensor for multiplication.\n");
        return NULL;
    }
    for (int i = 0; i < a -> size; i++) {
        result -> data[i] = a -> data[i] * b -> data[i];
    }
    if (result -> requires_grad) {
        result -> parents = (Tensor**)malloc(2 * sizeof(Tensor*));
        if (result -> parents == NULL) {
            fprintf(stderr, "Error: Failed to allocate memory for parents array in multiplication result tensor.\n");
            free_tensor(result);
            return NULL;
        }
        // Set the parents and operation type for autograd
        result -> parents[0] = a;
        result -> parents[1] = b;
        result -> num_parents = 2;
        result -> op = OP_MUL;
    }
    return result;
}

Tensor* tensor_matmul(Tensor *a, Tensor *b) {
    if (a -> ndims != 2 || b -> ndims != 2) {
        fprintf(stderr, "Error: Both tensors must be 2D for matrix multiplication.\n");
        return NULL;
    }
    if (a -> shape[1] != b -> shape[0]) {
        fprintf(stderr, "Error: Inner dimensions of tensors do not match for matrix multiplication.\n");
        return NULL;
    }

    int result_shape[] = {a -> shape[0], b -> shape[1]};
    Tensor* result = create_tensor(result_shape, 2, a -> requires_grad || b -> requires_grad);

    // Naive matrix multiplication implementation (for simplicity)
    for (int i = 0; i < a -> shape[0]; i ++) {
        for (int j = 0; j < b -> shape[1]; j ++) {
            float sum = 0.0f;
            for (int k = 0; k < a -> shape[1]; k ++) {
                sum += a -> data[i * a -> shape[1] + k] * b -> data[k * b -> shape[1] + j];
            }
            result -> data[i * result_shape[1] + j] = sum;
        }
    }

    if (result -> requires_grad) {
        result -> parents = (Tensor**)malloc(2 * sizeof(Tensor*));
        if (result -> parents == NULL) {
            fprintf(stderr, "Error: Failed to allocate memory for parents array in matrix multiplication result tensor.\n");
            free_tensor(result);
            return NULL;
        }
        // Set the parents and operation type for autograd
        result -> parents[0] = a;
        result -> parents[1] = b;
        result -> num_parents = 2;
        result -> op = OP_MATMUL;
    }
    return result;
}

Tensor* tensor_relu(Tensor* a) {
    // Apply ReLU activation function element-wise, return a new tensor
    Tensor* result = create_tensor(a -> shape, a -> ndims, a -> requires_grad);

    // Forward pass, max(0, x)
    for (int i = 0; i < a -> size; i++) {
        result -> data[i] = a -> data[i] > 0 ? a -> data[i] : 0.0f;
    }

    if (result -> requires_grad) {
        result -> parents = (Tensor**)malloc(sizeof(Tensor*));
        if (result -> parents == NULL) {
            fprintf(stderr, "Error: Failed to allocate memory for parents array in ReLU result tensor.\n");
            free_tensor(result);
            return NULL;
        }
        // Set the parent and operation type for autograd
        result -> parents[0] = a;
        result -> num_parents = 1;
        result -> op = OP_RELU;
    }
    return result;
}
