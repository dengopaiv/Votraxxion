"""Build the Python extension over the C DSP core.

The synthesizer itself is C, in src/; this builds the pybind11 shim in csrc/
against it so the Workbench GUI and the tests have a Python object to hold.
The NVDA add-on does not go through here at all -- see nvda-addon/package.py,
which compiles the same C sources straight to a DLL with no Python involved.
"""

from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

#: The chip half of the C core.  The letter-to-sound tables are not here:
#: the Python side does its own text handling and would only be paying for
#: 40 KB of string tables it never reads.
C_SOURCES = [
    "src/votrax_core.c",
    "src/votrax_filters.c",
    "src/votrax_rom.c",
]

ext_modules = [
    Pybind11Extension(
        "pyvotrax._votrax_core",
        sources=["csrc/bindings.cpp"] + C_SOURCES,
        include_dirs=["src", "csrc"],
        cxx_std=17,
    ),
]

setup(
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
)
