#!/bin/bash

compiler="g++" 	# "", "g++", "g++-4.9", "clang++"

build_type=Release
build_sys=ninja

install_dir="`pwd`/install"
install=OFF

bash build.sh \
	--compiler ${compiler} \
	--build-dir ./build-release \
	--build-type ${build_type} \
	--build-sys ${build_sys} \
	--install-dir ${install_dir} \
	--install ${install} \
	--asan OFF --tsan OFF


