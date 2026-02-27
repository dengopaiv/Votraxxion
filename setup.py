"""Build script for the C++ DSP core extension module."""

from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

ext_modules = [
    Pybind11Extension(
        "pyvotrax._votrax_core",
        sources=["csrc/bindings.cpp"],
        include_dirs=["csrc"],
        cxx_std=17,
        define_macros=[("_USE_MATH_DEFINES", "1")],  # M_PI on MSVC
    ),
]

setup(
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
)
