#!/bin/bash

build_type=Debug
cmake_cuda=OFF
cmake_opencl=OFF

bash build.sh ./build-debug ${build_type} ${cmake_cuda} ${cmake_opencl} OFF OFF OFF

bash build.sh ./build-debug-asan ${build_type} ${cmake_cuda} ${cmake_opencl} ON OFF OFF
bash build.sh ./build-debug-lsan ${build_type} ${cmake_cuda} ${cmake_opencl} OFF ON OFF
bash build.sh ./build-debug-tsan ${build_type} ${cmake_cuda} ${cmake_opencl} OFF OFF ON

