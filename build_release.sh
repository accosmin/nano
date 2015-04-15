#!/bin/bash

build_type=Release
install_dir="`pwd`/install"
install=OFF

bash build.sh \
	--build-dir ./build-release \
	--build-type ${build_type} \
	--install-dir ${install_dir} \
	--install ${install} \
	--asan OFF --lsan OFF --tsan OFF


