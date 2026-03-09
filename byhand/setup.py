import os
import sys
from setuptools import setup, Extension
from torch.utils import cpp_extension

# Add -O3 for optimization and -DNDEBUG to disable assertions
extra_compile_args = ['-std=c++20', '-O3', '-DNDEBUG']

if sys.platform == 'darwin':
    extra_compile_args.append('-Wno-invalid-specialization')
    # Use Apple Clang on macOS to avoid std libc++ linking errors with PyTorch
    if not os.environ.get("CC"):
        os.environ["CC"] = "/usr/bin/clang"
    if not os.environ.get("CXX"):
        os.environ["CXX"] = "/usr/bin/clang++"

setup(name='_byhand_dataset',
      ext_modules=[
          cpp_extension.CppExtension(
              '_byhand_dataset',
              ['dataset.cpp', '../src/game/Position.cpp', '../src/game/Move.cpp', '../src/StringUtils.cpp', '../src/game/CreateThreats.cpp', '../src/game/Utils.cpp', '../src/game/Geometry.cpp'],
              extra_compile_args=extra_compile_args,
              include_dirs=['../src']
          )
      ],
      cmdclass={'build_ext': cpp_extension.BuildExtension})
