import torch
from torch.utils.cpp_extension import load


def speed_test():
    size = [1000, 1000]
    x_large = torch.rand(size=size, device="cuda", dtype=torch.float32)
    num_runs = 100

    def time_function(func, *args, **kwargs):
        torch.cuda.synchronize()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        for _ in range(10):
            _ = func(*args, **kwargs)

        start_event.record()
        for _ in range(num_runs):
            _ = func(*args, **kwargs)

        end_event.record()
        torch.cuda.synchronize()

        return start_event.elapsed_time(end_event) / num_runs

    print(f"\nSpeed Test (average of {num_runs} runs)\n")

    custom_softmax_time = time_function(functions.softmax, x_large)
    custom_softmax_reduce_time = time_function(functions.softmax_reduce, x_large)
    torch_softmax_time = time_function(torch.nn.functional.softmax, x_large, dim=1)

    print(
        f"{'Operation':<15} | {'Custom CUDA (ms)':<15} | {'Custom CUDA Reduce (ms)':<15} | {'PyTorch (ms)':<15} | Speedup"
    )
    print("-" * 85)
    print(
        f"{'Softmax':<15} | {custom_softmax_time:15.3f} | {custom_softmax_reduce_time:15.3f} | {torch_softmax_time:15.3f} | {torch_softmax_time/custom_softmax_time:5.1f}x | {torch_softmax_time/custom_softmax_reduce_time:5.1f}x"
    )


sources = ["bind.cpp", "softmax.cu", "softmax_reduce.cu"]
functions = load("functions", sources=sources, verbose=True)

x = torch.rand([10, 10], device="cuda", dtype=torch.float32)
print(f"Input: {x}\n")

y = functions.softmax(x)
print(f"CUDA Output: {y}\n")
print(f"Sum of softmax output: {torch.sum(y, dim=1)}\n")

y_reduce = functions.softmax_reduce(x)
print(f"CUDA Reduce Output: {y_reduce}\n")
print(f"Sum of softmax reduce output: {torch.sum(y_reduce, dim=1)}\n")

softmax_torch = torch.nn.functional.softmax(x, dim=1)
print(f"PyTorch Output: {softmax_torch}\n")
print(f"Sum of softmax output: {torch.sum(softmax_torch, dim=1)}\n")

speed_test()
