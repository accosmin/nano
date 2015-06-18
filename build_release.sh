#!/bin/bash

build_type=Release
build_sys=ninja
install_dir="`pwd`/install"
install=OFF

bash build.sh \
	--build-dir ./build-release \
	--build-type ${build_type} \
	--build-sys ${build_sys} \
	--install-dir ${install_dir} \
	--install ${install} \
	--asan OFF --tsan OFF


