import os
import glob
from setuptools import setup, find_packages

from torch.utils import cpp_extension
from torch_mlu.utils.cpp_extension import MLUExtension, BuildExtension

cpath = os.path.join(os.path.abspath(os.path.dirname(__file__)), "csrc")

sources = list(glob.glob(f"{cpath}/*.cpp")) + list(glob.glob(f"{cpath}/kernel/*.mlu"))

mlu_extension = MLUExtension(
                    name="libmlu_ext",
                    sources=sources,
                    include_dirs=[os.path.join(cpath, "include")],
                    extra_compile_args={
                        "cxx": [
                            "-O3",
                            "-Wl,-rpath", 
                            "-std=c++14",
                            ],
                        "cncc":[
                            "-O3",
                            "--bang-mlu-arch=mtp_372",
                            "--bang-mlu-arch=mtp_592",
                            ]
                    }
                )

setup(
    name="mlu_ext",
    version="0.1",
    packages=find_packages(),
    ext_modules=[mlu_extension],
    # package_dir={'': 'mlu_ext'},
    cmdclass={
        "build_ext": BuildExtension.with_options(no_python_abi_suffix=True)
    }
)
