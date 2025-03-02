#include <torch/extension.h>

// Function declaration for CUDA kernel
torch::Tensor prewitt_cuda_forward(torch::Tensor input);

// PyTorch binding
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    // binding prewitt cuda forward to the python forward function
    m.def("forward", &prewitt_cuda_forward, "Prewitt Filter forward pass (CUDA)");
}
