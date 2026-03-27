#include "../include/autograd.h"

// Functions for the tensor array

typedef struct {
    Tensor** array;
    int size;
    int capacity;
} TensorArray;

void tensor_array_init(TensorArray* ta, int initial_capacity) {
    // Initialize a dynamic array to hold tensors during the topological sort
    ta->size = 0;
    ta->capacity = initial_capacity;
    ta->array = (Tensor**)malloc(ta->capacity * sizeof(Tensor*));
    if (ta->array == NULL) {
        fprintf(stderr, "Error: Failed to allocate memory for TensorArray.\n");
        exit(EXIT_FAILURE);
    }
}

void tensor_array_append(TensorArray* ta, Tensor* t) {
    // If we hit the limit, double the capacity using realloc
    if (ta->size >= ta->capacity) {
        ta->capacity *= 2;
        ta->array = (Tensor**)realloc(ta->array, ta->capacity * sizeof(Tensor*));
        
        if (ta->array == NULL) {
            fprintf(stderr, "Fatal Error: realloc failed in tensor_array_append.\n");
            exit(1);
        }
    }
    // Append the new tensor pointer and increment size
    ta->array[ta->size++] = t;
}

void tensor_array_free(TensorArray *ta) {
    // Free the memory allocated for the dynamic array of tensor pointers
    free(ta->array);
    ta->size = 0;
    ta->capacity = 0;
}

// Functions for the topological sort

bool is_visited(Tensor* t, TensorArray* visited) {
    // Helper to check if a tensor has already been visited during the topological sort
    for (int i = 0; i < visited->size; i++) {
        if (visited->array[i] == t) {
            return true;
        }
    }
    return false;
}

void build_topo(Tensor* u, TensorArray* topo, TensorArray* visited) {
    // Helper to perform DFS and build the topological order of tensors for backpropagation
    if (!is_visited(u, visited)) {
        tensor_array_append(visited, u);

        for (int i = 0; i < u -> num_parents; i ++) {
            build_topo(u -> parents[i], topo, visited);
        }

        tensor_array_append(topo, u);
    }
}

// Functions for the backward pass

void backward_add(Tensor* t) {
    // Takes tensor t which is the result of an addition operation, and computes the gradients for its parents
    if (t -> op != OP_ADD || t -> num_parents != 2) {
        fprintf(stderr, "Error: backward_add called on a tensor that is not the result of an addition operation.\n");
        return;
    }
    Tensor* a = t -> parents[0];
    Tensor* b = t -> parents[1];
    if (a -> requires_grad) {
        for (int i = 0; i < a -> size; i++) {
            a -> grad[i] += t -> grad[i];
        }
    }
    if (b -> requires_grad) {
        for (int i = 0; i < b -> size; i++) {
            b -> grad[i] += t -> grad[i];
        }
    }
}

void backward_mul(Tensor* t) {
    // Takes tensor t which is the result of a multiplication operation, and computes the gradients for its parents
    if (t -> op != OP_MUL || t -> num_parents != 2) {
        fprintf(stderr, "Error: backward_mul called on a tensor that is not the result of a multiplication operation.\n");
        return;
    }
    Tensor* a = t -> parents[0];
    Tensor* b = t -> parents[1];
    if (a -> requires_grad) {
        for (int i = 0; i < a -> size; i++) {
            a -> grad[i] += b -> data[i] * t -> grad[i];
        }
    }
    if (b -> requires_grad) {
        for (int i = 0; i < b -> size; i++) {
            b -> grad[i] += a -> data[i] * t -> grad[i];
        }
    }
}

void backward_matmul(Tensor* t) {
    // Takes tensor t which is the result of a matrix multiplication operation, and computes the gradients for its parents
    if (t -> op != OP_MATMUL || t -> num_parents != 2) {
        fprintf(stderr, "Error: backward_matmul called on a tensor that is not the result of a matrix multiplication operation.\n");
        return;
    }
    Tensor* a = t -> parents[0];
    Tensor* b = t -> parents[1];

    int M = a->shape[0];
    int K = a->shape[1];
    int N = b->shape[1];
   
    // gradient of a is grad_output @ b^T, and gradient of b is a^T @ grad_output
    if (a -> requires_grad) {
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < K; j++) {
                float grad_a_ij = 0.0f;
                for (int n = 0; n < N; n++) {
                    grad_a_ij += t->grad[i * N + n] * b->data[j * N + n]; // grad_output[i, n] * b[j, n]
                }
                a->grad[i * K + j] += grad_a_ij;
            }
        }
    }

    if (b->requires_grad) {
        for (int k = 0; k < K; k++) {
            for (int j = 0; j < N; j++) {
                float grad_b_kj = 0.0f;
                for (int m = 0; m < M; m++) {
                    grad_b_kj += a->data[m * K + k] * t->grad[m * N + j]; // a[m, k] * grad_output[m, j]
                }
                b->grad[k * N + j] += grad_b_kj;
            }
        }
    }
}

void backward_relu(Tensor* t) {
    // Takes tensor t which is the result of a relu operation, and computes the gradients for its parent
    if (t -> op != OP_RELU || t -> num_parents != 1) {
        fprintf(stderr, "Error: backward_relu called on a tensor that is not the result of a relu operation.\n");
        return;
    }
    Tensor* a = t -> parents[0];
    if (a -> requires_grad) {
        for (int i = 0; i < t -> size; i++) {
            a -> grad[i] += a -> data[i] > 0 ? t -> grad[i] : 0.0f; // grad_input = grad_output if input > 0 else 0
        }
    }
}

void backward_add_bias(Tensor* t) {
    if (t -> op != OP_ADDBIAS || t -> num_parents != 2) {
        fprintf(stderr, "Error: backward_add_bias called on a tensor that is not the result of an add_bias operation.\n");
        return;
    }
    Tensor* a = t -> parents[0];
    Tensor* bias = t -> parents[1];

    int batch_size = a->shape[0];
    int features = a->shape[1];

    if (a -> requires_grad) {
        for (int i = 0; i < a -> size; i++) {
            a -> grad[i] += t -> grad[i];
        }
    }
    if (bias -> requires_grad) {
        for (int i = 0; i < batch_size; i++) {
            for (int j = 0; j < features; j++) {
                bias -> grad[j] += t -> grad[i * features + j]; // Sum over the batch dimension
            }
        }
    }
}

void backward(Tensor* t) {
    // Main backward function that performs a topological sort of the computation graph and calls the appropriate backward functions in order
    TensorArray topo;
    TensorArray visited;
    tensor_array_init(&topo, 16); // Start with capacity for 16 tensors in the topological order
    tensor_array_init(&visited, 16); // Start with capacity for 16 visited tensors

    build_topo(t, &topo, &visited);

    t->grad[0] = 1.0f; // Seed the final output gradient with 1.0 (since dz/dz = 1)

    // Now we have the topological order of tensors in topo, we can call the backward functions in reverse order
    for (int i = topo.size - 1; i >= 0; i--) {
        Tensor* current = topo.array[i];
        if (current -> op == OP_ADD) {
            backward_add(current);
        } else if (current -> op == OP_MUL) {
            backward_mul(current);
        } else if (current -> op == OP_MATMUL) {
            backward_matmul(current);
        } else if (current -> op == OP_RELU) {
            backward_relu(current);
        } else if (current -> op == OP_ADDBIAS) {
            backward_add_bias(current);
        }
    }
    // Free allocated memory
    tensor_array_free(&topo);
    tensor_array_free(&visited);
}