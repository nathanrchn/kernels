from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="swissai",
    ext_modules=[
        CUDAExtension(
            name="swissai",
            sources=["src/binding.cpp", "src/xielu.cu"],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": [
                    "-O3",
                    "--use_fast_math",
                    "--ptxas-options=-v",
                    "-lineinfo",
                ]
            }
        )
    ],
    cmdclass={
        "build_ext": BuildExtension
    }
)
