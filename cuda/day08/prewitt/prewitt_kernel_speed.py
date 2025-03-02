import cv2
import torch
import time
import numpy as np
import prewitt_cuda
from torch.profiler import profile, record_function, ProfilerActivity

# Load grayscale image.
image_path = "images/pokemon.jpg"
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Measure time for OpenCV Prewitt filter.
prewitt_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
prewitt_y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])

start_time = time.time()
prewitt_x_oc = cv2.filter2D(image, -1, prewitt_x)
prewitt_y_oc = cv2.filter2D(image, -1, prewitt_y)
gradient_magnitude = np.sqrt(prewitt_x_oc**2 + prewitt_y_oc**2)
opencv_time = time.time() - start_time
print(f"\nOpenCV Prewitt Execution Time: {opencv_time:.6f} seconds\n")

# Measure time for PyTorch CUDA implementation.
image_np = np.array(image).astype(np.float32)
input_tensor = torch.from_numpy(image_np).cuda()

start_time = time.time()
prewitt_x_torch = prewitt_cuda.forward(input_tensor)
torch_time = time.time() - start_time
print(f"\nPyTorch CUDA Execution Time: {torch_time:.6f} seconds\n")


# Profile the PyTorch CUDA implementation.
with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True
) as prof:
    with record_function("CUDA Prewitt Filter"):
        prewitt_cuda_output = prewitt_cuda.forward(input_tensor)

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
