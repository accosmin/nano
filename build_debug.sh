#!/bin/bash

build_type=Debug
cmake_cuda=OFF
cmake_opencl=OFF

bash build.sh --build-dir ./build-debug --build-type ${build_type} --cuda ${cmake_cuda} --opencl ${cmake_opencl} --asan OFF --lsan OFF --tsan OFF

# does not work with clang!
#bash build.sh --build-dir ./build-debug-asan --build-type ${build_type} --cuda ${cmake_cuda} --opencl ${cmake_opencl} --asan ON --lsan OFF --tsan OFF
#bash build.sh --build-dir ./build-debug-lsan --build-type ${build_type} --cuda ${cmake_cuda} --opencl ${cmake_opencl} --asan OFF --lsan ON --tsan OFF
#bash build.sh --build-dir ./build-debug-tsan --build-type ${build_type} --cuda ${cmake_cuda} --opencl ${cmake_opencl} --asan OFF --lsan OFF --tsan ON

