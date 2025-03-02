from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

setup(
    name="prewitt_cuda",
    ext_modules=[
        CUDAExtension("prewitt_cuda", ["prewitt.cpp", "prewitt_kernel.cu"]),
    ],
    cmdclass={"build_ext": BuildExtension},
)
