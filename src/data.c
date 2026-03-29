#include "../include/data.h"
#include <stdint.h>

uint32_t swap_endian(uint32_t val) {
    // Helper function to flip Big-Endian to Little-Endian
    return ((val << 24) & 0xFF000000) |
           ((val <<  8) & 0x00FF0000) |
           ((val >>  8) & 0x0000FF00) |
           ((val >> 24) & 0x000000FF);
}

Tensor* load_mnist_images(const char* filename) {
    // Loads the MNIST images from the binary file.
    // Normalizes the 0-255 pixel values into 0.0f - 1.0f floats.
    // Returns a Tensor of shape [num_images, 784]
    FILE* file = fopen(filename, "rb");
    if (!file) {
        fprintf(stderr, "Error: Could not open file %s\n", filename);
        return NULL;
    }

    uint32_t magic_number, num_images, num_rows, num_cols;
    
    // Read the 4-integer header
    fread(&magic_number, sizeof(magic_number), 1, file);
    fread(&num_images, sizeof(num_images), 1, file);
    fread(&num_rows, sizeof(num_rows), 1, file);
    fread(&num_cols, sizeof(num_cols), 1, file);

    // Flip the bytes to Little-Endian
    magic_number = swap_endian(magic_number);
    num_images = swap_endian(num_images);
    num_rows = swap_endian(num_rows);
    num_cols = swap_endian(num_cols);

    if (magic_number != 2051) {
        fprintf(stderr, "Invalid MNIST image file magic number.\n");
        fclose(file);
        return NULL;
    }

    int features = num_rows * num_cols; // 28 * 28 = 784
    int shape[] = {num_images, features};
    Tensor* images = create_tensor(shape, 2, false);

    // Read the raw pixel bytes (0-255)
    uint8_t* raw_pixels = (uint8_t*)malloc(num_images * features * sizeof(uint8_t));
    fread(raw_pixels, sizeof(uint8_t), num_images * features, file);

    // Convert to floats and normalize to 0.0 - 1.0
    for (int i = 0; i < images->size; i++) {
        images->data[i] = (float)raw_pixels[i] / 255.0f;
    }

    free(raw_pixels);
    fclose(file);
    return images;
}

Tensor* load_mnist_labels(const char* filename) {
    // Loads the MNIST labels from the binary file.
    // Converts the raw digits (0-9) into one-hot encoded vectors.
    // Returns a Tensor of shape [num_items, 10]
    FILE* file = fopen(filename, "rb");
    if (!file) {
        fprintf(stderr, "Error: Could not open file %s\n", filename);
        return NULL;
    }

    uint32_t magic_number, num_items;
    
    // Read the 2-integer header
    fread(&magic_number, sizeof(magic_number), 1, file);
    fread(&num_items, sizeof(num_items), 1, file);

    magic_number = swap_endian(magic_number);
    num_items = swap_endian(num_items);

    if (magic_number != 2049) {
        fprintf(stderr, "Invalid MNIST label file magic number.\n");
        fclose(file);
        return NULL;
    }

    // We create a one-hot encoded matrix: [num_items, 10]
    int shape[] = {num_items, 10};
    Tensor* labels = create_tensor(shape, 2, false);

    // Initialize all to 0.0f
    for (int i = 0; i < labels->size; i++) {
        labels->data[i] = 0.0f;
    }

    // Read the raw labels (0-9)
    uint8_t* raw_labels = (uint8_t*)malloc(num_items * sizeof(uint8_t));
    fread(raw_labels, sizeof(uint8_t), num_items, file);

    // Set the specific index to 1.0f for the one-hot encoding
    for (int i = 0; i < num_items; i++) {
        uint8_t digit = raw_labels[i];
        labels->data[i * 10 + digit] = 1.0f; 
    }

    free(raw_labels);
    fclose(file);
    return labels;
}