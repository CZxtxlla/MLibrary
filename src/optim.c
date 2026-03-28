#include "../include/optim.h"

SGD* sgd_create(Tensor** parameters, int num_parameters, float lr) {
    // Allocates the optimizer and stores the pointers to the learnable parameters
    SGD* optim = (SGD*)malloc(sizeof(SGD));
    if (optim == NULL) {
        fprintf(stderr, "Error: Failed to allocate memory for SGD optimizer.\n");
        return NULL;
    }
    optim->parameters = parameters;
    optim->num_parameters = num_parameters;
    optim->lr = lr;

    return optim;
}

void sgd_step(SGD* optim) {
    // Loops through all parameters and applies the learning rule: data = data - (lr * grad)
    for (int i = 0; i < optim->num_parameters; i++) {
        Tensor* param = optim->parameters[i];
        if (param->requires_grad && param->grad != NULL) {
            for (int j = 0; j < param->size; j++) {
                param->data[j] -= optim->lr * param->grad[j];
            }
        }
    }
}

void sgd_zero_grad(SGD* optim) {
    // Loop through every tensor and clear the gradients
    for (int i = 0; i < optim->num_parameters; i++) {
        Tensor* p = optim->parameters[i];
        
        if (p->requires_grad) {
            for (int j = 0; j < p->size; j++) {
                // If we don't do this, gradients will accumulate infinitely epoch after epoch
                p->grad[j] = 0.0f;
            }
        }
    }
}

void sgd_free(SGD* optim) {
    // Frees the optimizer struct itself (but not the parameters it points to)
    if (optim != NULL) {
        free(optim);
    }
}
