import time
from mlibrary import Tensor, MLP, SGD, load_mnist_images, load_mnist_labels, fetch_batch, cross_entropy_loss

def calculate_accuracy(model, dataset_x, dataset_y, batch_size=64):
    num_samples = dataset_x.shape[0]
    num_batches = num_samples // batch_size
    correct = 0
    
    # Pre-allocate temporary batch tensors
    batch_x = Tensor.empty([batch_size, 784])
    batch_y = Tensor.empty([batch_size, 10])
    
    for step in range(num_batches):
        start_idx = step * batch_size
        
        # 1. C loads the data
        fetch_batch(dataset_x, dataset_y, start_idx, batch_size, batch_x, batch_y)
        
        # 2. C runs the heavy matrix multiplication
        logits = model(batch_x)
        
        # 3. Pull the results into Python to find the argmax
        l_data = logits.get_data()
        t_data = batch_y.get_data()
        
        for i in range(batch_size):
            # Isolate the 10 predictions and 10 targets for this specific image
            start = i * 10
            end = start + 10
            image_logits = l_data[start:end]
            image_targets = t_data[start:end]
            
            # Find the index of the highest value
            pred_class = image_logits.index(max(image_logits))
            true_class = image_targets.index(max(image_targets))
            
            if pred_class == true_class:
                correct += 1
                
        # Clean up the C memory for this batch
        logits.free_graph()
        
    return (correct / (num_batches * batch_size)) * 100.0

if __name__ == "__main__":
    print("=== MLibrary (Python API) MNIST Training ===")

    # 1. Load Data
    print("Loading datasets...")
    train_images = load_mnist_images("data/train-images-idx3-ubyte")
    train_labels = load_mnist_labels("data/train-labels-idx1-ubyte")
    
    num_samples = train_images.shape[0]
    print(f"Loaded {num_samples} images.\n")

    # 2. Setup Model & Optimizer
    model = MLP([784, 128, 10])
    optim = SGD(model, lr=0.05)

    # 3. Pre-allocate Mini-Batch Tensors
    batch_size = 64
    batch_x = Tensor.empty([batch_size, 784])
    batch_y = Tensor.empty([batch_size, 10])

    # 4. The Training Loop
    epochs = 3
    steps_per_epoch = num_samples // batch_size

    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        start_time = time.time()
        
        print(f"Epoch {epoch}/{epochs}")

        for step in range(steps_per_epoch):
            start_idx = step * batch_size
            
            # C copies the data behind the scenes
            fetch_batch(train_images, train_labels, start_idx, batch_size, batch_x, batch_y)
            
            optim.zero_grad()
            
            # Forward Pass
            logits = model(batch_x)
            
            # Loss & Gradients
            loss = cross_entropy_loss(logits, batch_y)
            epoch_loss += loss
            
            # Backward & Step
            logits.backward()
            optim.step()
            
            logits.free_graph()

            if step % 100 == 0 or step == steps_per_epoch - 1:
                print(f"\r  -> Step {step:4d}/{steps_per_epoch:4d} | Current Batch Loss: {loss:8.4f}", end="")

        time_spent = time.time() - start_time
        print(f"\n  [Summary] Average Loss: {epoch_loss / steps_per_epoch:8.4f} | Time: {time_spent:.2f}s\n")
        print("Evaluating Model Accuracy on Training Data...")
        accuracy = calculate_accuracy(model, train_images, train_labels, batch_size)
        print(f"  Training Accuracy: {accuracy:.2f}%\n")

    print("Training Complete!")