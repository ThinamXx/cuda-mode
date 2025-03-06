import torch
from torch.utils.cpp_extension import load

sources = ["bind.cpp", "softmax.cu"]
functions = load("functions", sources=sources, verbose=True)

x = torch.rand([10, 10], device="cuda", dtype=torch.float32)
print(f"Input: {x}\n")

y = functions.softmax(x)
print(f"CUDA Output: {y}\n")
print(f"Sum of softmax output: {torch.sum(y, dim=1)}\n")

softmax_torch = torch.nn.functional.softmax(x, dim=1)
print(f"PyTorch Output: {softmax_torch}\n")
print(f"Sum of softmax output: {torch.sum(softmax_torch, dim=1)}\n")

