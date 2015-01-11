#!/bin/bash

build_type=Release
cmake_cuda=OFF
cmake_opencl=OFF

bash build.sh --build-dir ./build-release --install-dir "/usr/local/" --build-type ${build_type} --cuda ${cmake_cuda} --opencl ${cmake_opencl} --asan OFF --lsan OFF --tsan OFF


