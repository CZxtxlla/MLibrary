#ifndef DATA_H
#define DATA_H

#include "tensor.h"

// Loads the MNIST images from the binary file.
// Normalizes the 0-255 pixel values into 0.0f - 1.0f floats.
// Returns a Tensor of shape [num_images, 784].
Tensor* load_mnist_images(const char* filename);

// Loads the MNIST labels from the binary file.
// Converts the raw digits (0-9) into one-hot encoded vectors.
// Returns a Tensor of shape [num_items, 10].
Tensor* load_mnist_labels(const char* filename);

#endif // DATA_H