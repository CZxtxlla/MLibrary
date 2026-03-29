#ifndef LOSS_H
#define LOSS_H

#include "tensor.h"

// Computes Softmax + Cross Entropy Loss across a batch.
// - logits: The raw [batch_size, num_classes] output from your MLP.
// - targets: A [batch_size, num_classes] tensor of one-hot encoded true labels.
float cross_entropy_loss(Tensor* logits, Tensor* targets);

#endif 