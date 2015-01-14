#!/bin/bash

build_type=Release
install_dir="/usr/local/"
install=OFF

cmake_cuda=OFF
cmake_opencl=OFF

bash build.sh \
	--build-dir ./build-release \
	--build-type ${build_type} \
	--install-dir ${install_dir} \
	--install ${install} \
	--cuda ${cmake_cuda} \
	--opencl ${cmake_opencl} \
	--asan OFF --lsan OFF --tsan OFF


