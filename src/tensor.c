#include "../include/tensor.h"
#include "../include/cuda_backend.h"


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
    t->device = DEVICE_CPU; // Everything starts on the CPU by default
    t->gpu_data = NULL;     // Explicitly set to NULL so we know it's empty
    t->gpu_grad = NULL;
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
    
    if(t->gpu_data != NULL || t->gpu_grad != NULL) {
        cuda_free_device_memory(t); // For GPU memory, must implement cuda_free in GPU utility file
    }
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

Tensor* tensor_add_bias(Tensor* a, Tensor* bias) {
    // Safety check: 'a' should be 2D [Batch, Features], 'bias' should be [1, Features]
    if (a->shape[1] != bias->shape[1]) {
        fprintf(stderr, "Fatal Error: Bias broadcasting dimension mismatch.\n");
        return NULL;
    }

    int batch_size = a->shape[0];
    int features = a->shape[1];

    Tensor* out = create_tensor(a->shape, a->ndims, a->requires_grad || bias->requires_grad);
    // Forward pass: add the bias to every row
    for (int i = 0; i < batch_size; i++) {
        for (int j = 0; j < features; j++) {
            // out[i, j] = a[i, j] + bias[0, j]
            out->data[i * features + j] = a->data[i * features + j] + bias->data[j];
        }
    }

    // Link the autograd graph
    if (out->requires_grad) {
        out->op = OP_ADDBIAS; // Let's assume 5 is OP_ADD_BIAS. (Update your enum/defines if you have them!)
        out->num_parents = 2;
        out->parents = (Tensor**)malloc(2 * sizeof(Tensor*));
        out->parents[0] = a;
        out->parents[1] = bias;
    }

    return out;
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

void tensor_to_device(Tensor* t, DeviceType device) {
    // transfers tensor data and gradients between CPU and GPU memory.
    if (t->device == device) {
        return; // Already on the desired device
    }
    if (device == DEVICE_CPU) {
        cuda_copy_to_host(t); // Copies data from GPU to CPU
        t-> device = DEVICE_CPU;
    } else if (device == DEVICE_GPU) {
        cuda_copy_to_device(t); // Copies data from CPU to GPU
        t-> device = DEVICE_GPU;
    } else {
        fprintf(stderr, "Error: Invalid device type specified for tensor transfer.\n");
    }
}

