#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "../include/tensor.h"
#include "../include/nn.h"
#include "../include/autograd.h"
#include "../include/optim.h"
#include "../include/loss.h"
#include "../include/data.h"

// Helper function to copy a chunk of the dataset into our active batch tensors
void fetch_batch(Tensor* dataset_x, Tensor* dataset_y, int start_idx, int batch_size, Tensor* batch_x, Tensor* batch_y) {
    int features = dataset_x->shape[1]; // 784
    int classes = dataset_y->shape[1];  // 10
    
    for (int i = 0; i < batch_size; i++) {
        for (int j = 0; j < features; j++) {
            batch_x->data[i * features + j] = dataset_x->data[(start_idx + i) * features + j];
        }
        for (int j = 0; j < classes; j++) {
            batch_y->data[i * classes + j] = dataset_y->data[(start_idx + i) * classes + j];
        }
    }
}

// Evaluates the network's accuracy by checking how many predictions match the targets
float calculate_accuracy(LinearLayer* layer1, LinearLayer* layer2, Tensor* dataset_x, Tensor* dataset_y, int batch_size) {
    int num_samples = dataset_x->shape[0];
    int num_batches = num_samples / batch_size;
    int correct_count = 0;

    // We need temporary tensors for the batch to avoid massive memory allocation
    int batch_x_shape[] = {batch_size, 784};
    int batch_y_shape[] = {batch_size, 10};
    Tensor* batch_x = create_tensor(batch_x_shape, 2, false);
    Tensor* batch_y = create_tensor(batch_y_shape, 2, false);

    for (int step = 0; step < num_batches; step++) {
        int start_idx = step * batch_size;
        fetch_batch(dataset_x, dataset_y, start_idx, batch_size, batch_x, batch_y);

        // Forward Pass (Making our predictions)
        Tensor* z1 = linear_forward(layer1, batch_x);
        Tensor* a1 = tensor_relu(z1);
        Tensor* logits = linear_forward(layer2, a1);

        // Compare predictions to true targets
        for (int i = 0; i < batch_size; i++) {
            float max_logit = -INFINITY;
            int pred_class = -1;
            
            float max_target = -INFINITY;
            int true_class = -1;

            // Argmax: Find the index of the highest probability and highest target
            for (int c = 0; c < 10; c++) {
                float logit = logits->data[i * 10 + c];
                if (logit > max_logit) {
                    max_logit = logit;
                    pred_class = c;
                }

                float target = batch_y->data[i * 10 + c];
                if (target > max_target) {
                    max_target = target;
                    true_class = c;
                }
            }

            // Did the network guess right?
            if (pred_class == true_class) {
                correct_count++;
            }
        }

        // Clean up the computation graph for this batch
        free_graph(logits);
    }

    free_tensor(batch_x);
    free_tensor(batch_y);

    // Return the percentage of correct predictions
    return ((float)correct_count / (float)(num_batches * batch_size)) * 100.0f;
}

int main() {
    printf("=== MLibrary MNIST Training ===\n\n");

    // 1. Load the Dataset (Make sure you ran the curl commands to download these!)
    printf("Loading MNIST Dataset...\n");
    Tensor* train_images = load_mnist_images("data/train-images-idx3-ubyte");
    Tensor* train_labels = load_mnist_labels("data/train-labels-idx1-ubyte");
    
    int num_samples = train_images->shape[0];
    printf("Successfully loaded %d training images and labels.\n\n", num_samples);

    // --- Load the Test Dataset ---
    Tensor* test_images = load_mnist_images("data/t10k-images-idx3-ubyte");
    Tensor* test_labels = load_mnist_labels("data/t10k-labels-idx1-ubyte");
    printf("Successfully loaded %d test images.\n\n", test_images->shape[0]);

    // 2. Setup the Architecture (784 Pixels -> 128 Hidden -> 10 Digit Classes)
    int architecture[] = {784, 128, 10}; 
    //MLP* model = create_mlp(architecture, 3); // 3 layers: Input, Hidden, Output

    LinearLayer* layer1 = create_linear_layer(architecture[0], architecture[1]);
    LinearLayer* layer2 = create_linear_layer(architecture[1], architecture[2]);


    // 3. Setup the Optimizer (SGD)
    int num_params = 4;
    Tensor* params[] = {layer1->weight, layer1->bias, layer2->weight, layer2->bias}; 
    SGD* optim = sgd_create(params, num_params, 0.05f); // Learning rate = 0.05

    // 4. Pre-allocate the Mini-Batch Tensors
    // We allocate these ONCE to save memory, and just overwrite their data in the loop
    int batch_size = 64;
    int batch_x_shape[] = {batch_size, 784};
    int batch_y_shape[] = {batch_size, 10};
    Tensor* batch_x = create_tensor(batch_x_shape, 2, false);
    Tensor* batch_y = create_tensor(batch_y_shape, 2, false);

    printf("Starting Training Loop...\n");
    
    // ==========================================
    // THE MINI-BATCH TRAINING LOOP
    // ==========================================
    int epochs = 3; 
    int steps_per_epoch = num_samples / batch_size;

    for (int epoch = 1; epoch <= epochs; epoch++) {
        float epoch_loss = 0.0f;
        
        // Track time to calculate iterations per second
        clock_t start_time = clock();
        
        printf("Epoch %d/%d\n", epoch, epochs);

        for (int step = 0; step < steps_per_epoch; step++) {
            
            // Step A: Load the next 64 images into our batch tensors
            int start_idx = step * batch_size;
            fetch_batch(train_images, train_labels, start_idx, batch_size, batch_x, batch_y);

            // Step B: Zero Gradients
            sgd_zero_grad(optim);

            // Step C: Forward Pass
            Tensor* z1 = linear_forward(layer1, batch_x);
            Tensor* a1 = tensor_relu(z1);
            Tensor* logits = linear_forward(layer2, a1);

            // Step D: Calculate Loss & Inject Gradients
            float loss = cross_entropy_loss(logits, batch_y);
            epoch_loss += loss;

            // Step E: Backward Pass
            backward(logits);

            // Step F: Optimizer Step
            sgd_step(optim);

            // Step G: Garbage Collection
            free_graph(logits);

            // --- NEW: Live Progress Printing ---
            // Print progress every 100 steps, or on the very last step
            if (step % 100 == 0 || step == steps_per_epoch - 1) {
                // \r moves the cursor back to the start of the line. 
                // fflush forces it to draw immediately rather than waiting for a \n
                printf("\r  -> Step %4d/%4d | Current Batch Loss: %8.4f", step, steps_per_epoch, loss);
                fflush(stdout); 
            }
        }

        // Print the final summary for the epoch on a new line
        clock_t end_time = clock();
        double time_spent = (double)(end_time - start_time) / CLOCKS_PER_SEC;

        printf("\n  [Summary] Average Loss: %8.4f | Time: %.2fs\n\n", 
               epoch_loss / steps_per_epoch, time_spent);
    }

    printf("\nTraining Complete!\n");

    // --- Calculate Final Accuracy ---
    printf("Evaluating Model Accuracy on Test Data...\n");
    float accuracy = calculate_accuracy(layer1, layer2, test_images, test_labels, batch_size);
    printf("Final Test Accuracy: %.2f%%\n\n", accuracy);

    // 5. Memory Cleanup
    free_tensor(train_images);
    free_tensor(train_labels);
    free_tensor(test_images);
    free_tensor(test_labels);
    free_tensor(batch_x);
    free_tensor(batch_y);
    free_linear_layer(layer1);
    free_linear_layer(layer2);
    sgd_free(optim);

    return 0;
}