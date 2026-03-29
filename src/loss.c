#include "../include/loss.h"

float cross_entropy_loss(Tensor* logits, Tensor* targets){
    // Computes Softmax + Cross Entropy Loss across a batch.

    if (logits->shape[0] != targets->shape[0] || logits->shape[1] != targets->shape[1]) {
        fprintf(stderr, "Fatal Error: Logits and Targets shape mismatch in Cross Entropy.\n");
        exit(1);
    }

    int batch_size = logits->shape[0];
    int num_classes = logits->shape[1];
    float total_loss = 0.0f;

    for (int b = 0; b < batch_size; b++) {
        // Compute max logit so we can do stable softmax
        float max_logit = logits->data[b * num_classes];
        for (int c = 1; c < num_classes; c++) {
            float logit = logits->data[b * num_classes + c];
            if (logit > max_logit) {
                max_logit = logit;
            }
        }
        // Compute the sum of exp(logits - max_logit) for the denominator of softmax
        float sum_exp = 0.0f;
        float exps[num_classes]; // Store the exponentials for later use
        for (int c = 0; c < num_classes; c++) {
            exps[c] = expf(logits->data[b * num_classes + c] - max_logit);
            sum_exp += exps[c];
        }

        // Compute probabilities
        for (int i = 0; i < num_classes; i++) {
            float prob = exps[i] / sum_exp; // Softmax probability for class i
            float target = targets->data[b * num_classes + i];
            
            // Accumulate loss for the correct class
            if (target > 0.5f) { // This is the 1 in the one-hot vector
                // cross entropy loss for this example is -log(prob of the correct class)
                total_loss += -logf(prob + 1e-7f); // Add a tiny epsilon (1e-7) to prevent log(0)
                // this is a += because of multiple batches, but in a single example scenario this is just total_loss = -log(prob)
            }

            // The beautifully simple fused gradient: (Prob - Target)
            // We divide by batch_size so the gradients don't explode with large batches
            logits->grad[b * num_classes + i] = (prob - target) / (float)batch_size;
        }
    }
    return total_loss / (float)batch_size; // Return the average loss per example

}