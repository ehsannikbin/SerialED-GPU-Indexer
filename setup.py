from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='pinkindexer_cuda',
    ext_modules=[
        CUDAExtension('pinkindexer_cuda', [
            'csrc/bindings.cpp',        # <-- Added
            'csrc/rotogram.cpp',        # <-- Added
            'csrc/rotogram_kernel.cu',  # <-- Added
            'csrc/refiner.cpp',         # <-- Added
        ])
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)