import torch

a = torch.tensor([1.0, 2.0, 3.0])

print(torch.square(a))
print(a**2)
print(a * a)


def time_pytorch_function(func, input):
    # CUDA IS ASYNC so can't use python time module
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    # Warmup
    for _ in range(5):
        func(input)

    start.record()
    func(input)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end)


b = torch.randn(10000, 10000).cuda()


def square_2(a):
    return a * a


def square_3(a):
    return a**2


def cube(a):
    return a * a * a


time_pytorch_function(torch.square, b)
time_pytorch_function(square_2, b)
time_pytorch_function(square_3, b)

print("=============")
print("Profiling torch.square")
print("=============")

# Now profile each function using pytorch profiler
with torch.autograd.profiler.profile(use_device="cuda") as prof:
    torch.square(b)

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

print("=============")
print("Profiling a * a")
print("=============")

with torch.autograd.profiler.profile(use_device="cuda") as prof:
    square_2(b)

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

print("=============")
print("Profiling a ** 2")
print("=============")

with torch.autograd.profiler.profile(use_device="cuda") as prof:
    square_3(b)

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

print("=============")
print("Profiling a * a * a")
print("=============")

with torch.autograd.profiler.profile(use_device="cuda") as prof:
    cube(b)

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
