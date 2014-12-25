#!/bin/bash

build_type=Release
cmake_cuda=OFF
cmake_opencl=OFF

bash build.sh ./build-release ${build_type} ${cmake_cuda} ${cmake_opencl} OFF OFF OFF


