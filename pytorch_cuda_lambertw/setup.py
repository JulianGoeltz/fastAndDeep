from setuptools import setup, Extension
from torch.utils import cpp_extension

setup(
    name="lambertw_cuda",
    ext_modules=[
        cpp_extension.CUDAExtension(
            name="lambertw_cuda",
            sources=["lambertw_cuda.cpp", "plog_wrapper.cu"],
            extra_compile_args={'cxx': ['-g'],
                                'nvcc': ['-O2']}),
    ],
    cmdclass={"build_ext": cpp_extension.BuildExtension},
)
