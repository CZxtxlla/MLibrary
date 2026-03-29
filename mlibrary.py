import ctypes
import os

# ==========================================
# Load the Library & Define C-Structs
# ==========================================
lib_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "libmlibrary.so"))
lib = ctypes.CDLL(lib_path)

class CTensor(ctypes.Structure): pass
CTensor._fields_ = [
    ("data", ctypes.POINTER(ctypes.c_float)),
    ("grad", ctypes.POINTER(ctypes.c_float)),
    ("shape", ctypes.POINTER(ctypes.c_int)),
    ("ndims", ctypes.c_int),
    ("size", ctypes.c_int),
    ("requires_grad", ctypes.c_bool),
    ("op", ctypes.c_int),
    ("parents", ctypes.POINTER(ctypes.POINTER(CTensor))),
    ("num_parents", ctypes.c_int)
]

class CMLP(ctypes.Structure): pass
class CSGD(ctypes.Structure): pass

# Setup strict argtypes/restypes for safety
lib.create_tensor.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_bool]
lib.create_tensor.restype = ctypes.POINTER(CTensor)
lib.free_tensor.argtypes = [ctypes.POINTER(CTensor)]

lib.create_mlp.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int]
lib.create_mlp.restype = ctypes.POINTER(CMLP)
lib.mlp_forward.argtypes = [ctypes.POINTER(CMLP), ctypes.POINTER(CTensor)]
lib.mlp_forward.restype = ctypes.POINTER(CTensor)
lib.free_mlp.argtypes = [ctypes.POINTER(CMLP)]

lib.mlp_get_parameters.argtypes = [ctypes.POINTER(CMLP), ctypes.POINTER(ctypes.c_int)]
lib.mlp_get_parameters.restype = ctypes.POINTER(ctypes.POINTER(CTensor))

lib.sgd_create.argtypes = [ctypes.POINTER(ctypes.POINTER(CTensor)), ctypes.c_int, ctypes.c_float]
lib.sgd_create.restype = ctypes.POINTER(CSGD)
lib.sgd_zero_grad.argtypes = [ctypes.POINTER(CSGD)]
lib.sgd_step.argtypes = [ctypes.POINTER(CSGD)]
lib.sgd_free.argtypes = [ctypes.POINTER(CSGD)]

lib.backward.argtypes = [ctypes.POINTER(CTensor)]
lib.free_graph.argtypes = [ctypes.POINTER(CTensor)]

# --- Data & Loss API ---
lib.load_mnist_images.argtypes = [ctypes.c_char_p]
lib.load_mnist_images.restype = ctypes.POINTER(CTensor)
lib.load_mnist_labels.argtypes = [ctypes.c_char_p]
lib.load_mnist_labels.restype = ctypes.POINTER(CTensor)

lib.fetch_batch.argtypes = [
    ctypes.POINTER(CTensor), ctypes.POINTER(CTensor), 
    ctypes.c_int, ctypes.c_int, 
    ctypes.POINTER(CTensor), ctypes.POINTER(CTensor)
]

lib.cross_entropy_loss.argtypes = [ctypes.POINTER(CTensor), ctypes.POINTER(CTensor)]
lib.cross_entropy_loss.restype = ctypes.c_float


# ==========================================
# Pythonic API Wrappers
# ==========================================

class Tensor:
    def __init__(self, data_list, shape, requires_grad=False):
        flat_data = data_list
        if isinstance(data_list[0], list):
            flat_data = [item for sublist in data_list for item in sublist]
            
        c_shape = (ctypes.c_int * len(shape))(*shape)
        self.c_ptr = lib.create_tensor(c_shape, len(shape), requires_grad)
        
        for i, val in enumerate(flat_data):
            self.c_ptr.contents.data[i] = float(val)

    # --- empty tensor allocation ---
    @classmethod
    def empty(cls, shape, requires_grad=False):
        c_shape = (ctypes.c_int * len(shape))(*shape)
        c_ptr = lib.create_tensor(c_shape, len(shape), requires_grad)
        
        # Bypass __init__ and just wrap the pointer
        obj = cls.__new__(cls)
        obj.c_ptr = c_ptr
        return obj

    # --- wrap an existing C pointer from dataloader ---
    @classmethod
    def from_ptr(cls, c_ptr):
        obj = cls.__new__(cls)
        obj.c_ptr = c_ptr
        return obj

    # --- Get the shape as a Python list ---
    @property
    def shape(self):
        ndims = self.c_ptr.contents.ndims
        return [self.c_ptr.contents.shape[i] for i in range(ndims)]

    def __del__(self):
        if getattr(self, 'c_ptr', None):
            lib.free_tensor(self.c_ptr)
            self.c_ptr = None
            
    def get_data(self):
        size = self.c_ptr.contents.size
        return [self.c_ptr.contents.data[i] for i in range(size)]
        
    def backward(self):
        lib.backward(self.c_ptr)
        
    def free_graph(self):
        if getattr(self, 'c_ptr', None):
            lib.free_graph(self.c_ptr)
            self.c_ptr = None

class MLP:
    def __init__(self, architecture):
        c_arch = (ctypes.c_int * len(architecture))(*architecture)
        self.c_ptr = lib.create_mlp(c_arch, len(architecture))
        
    def __call__(self, x_tensor):
        # Override the function call operator (e.g., model(x))
        out_ptr = lib.mlp_forward(self.c_ptr, x_tensor.c_ptr)
        
        # Wrap the returned C pointer in a dummy Python Tensor so we can use it
        out_tensor = Tensor.__new__(Tensor)
        out_tensor.c_ptr = out_ptr
        return out_tensor
        
    def parameters(self):
        num_params = ctypes.c_int(0)
        c_params_array = lib.mlp_get_parameters(self.c_ptr, ctypes.byref(num_params))
        return c_params_array, num_params.value

    def __del__(self):
        if getattr(self, 'c_ptr', None):
            lib.free_mlp(self.c_ptr)

class SGD:
    def __init__(self, model, lr=0.01):
        params_array, num_params = model.parameters()
        self.c_ptr = lib.sgd_create(params_array, num_params, float(lr))
        
    def zero_grad(self):
        lib.sgd_zero_grad(self.c_ptr)
        
    def step(self):
        lib.sgd_step(self.c_ptr)
        
    def __del__(self):
        if getattr(self, 'c_ptr', None):
            lib.sgd_free(self.c_ptr)

# --- Functional API (Like torch.nn.functional) ---
def load_mnist_images(path):
    c_ptr = lib.load_mnist_images(path.encode('utf-8'))
    return Tensor.from_ptr(c_ptr)

def load_mnist_labels(path):
    c_ptr = lib.load_mnist_labels(path.encode('utf-8'))
    return Tensor.from_ptr(c_ptr)

def fetch_batch(dataset_x, dataset_y, start_idx, batch_size, batch_x, batch_y):
    # Pass the raw C pointers down to the engine
    lib.fetch_batch(dataset_x.c_ptr, dataset_y.c_ptr, start_idx, batch_size, batch_x.c_ptr, batch_y.c_ptr)

def cross_entropy_loss(logits, targets):
    # Computes loss and injects gradients natively in C
    return lib.cross_entropy_loss(logits.c_ptr, targets.c_ptr)

# ==========================================
# Training loop and example usage
# ==========================================
if __name__ == "__main__":
    print("--- MLibrary Python API ---")

    # 1. Setup Data (Clean Python Lists!)
    x = Tensor(
        data_list=[[0,0], [0,1], [1,0], [1,1]], 
        shape=[4, 2]
    )
    targets = [0.0, 1.0, 1.0, 0.0]

    # 2. Setup Model & Optimizer
    model = MLP([2, 8, 1])
    optim = SGD(model, lr=0.1)

    print("Training XOR...")

    # 3. The Training Loop
    for epoch in range(1, 500):
        optim.zero_grad()
        
        # forward pass
        out = model(x)
        
        # Calculate MSE Loss and inject gradients
        total_loss = 0.0
        for i in range(4):
            pred = out.get_data()[i]
            diff = pred - targets[i]
            total_loss += diff * diff
            # Inject the gradient of the loss w.r.t. the output
            out.c_ptr.contents.grad[i] = 2.0 * diff
            
        out.backward()
        optim.step()
        out.free_graph()

        if epoch % 20 == 0:
            print(f"Epoch {epoch:4d} | Loss: {total_loss / 4.0:.4f}")

    print("\nFinal Predictions:")
    final_out = model(x)
    preds = final_out.get_data()
    for i in range(4):
        print(f"Target: {targets[i]} | Prediction: {preds[i]:.4f}")
        
    final_out.free_graph()