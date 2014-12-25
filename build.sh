#!/bin/bash

if [ $# -ne 7 ]
then
	echo "Usage: "
	echo -e "\t<build directory>" 
	echo -e "\t<build type [Release/Debug]>"
	echo -e "\t<CUDA flag [ON/OFF]>"
	echo -e "\t<OpenCL flag [ON/OFF]>"
	echo -e "\t<address sanitizer [ON/OFF]>"
	echo -e "\t<leak sanitizer [ON/OFF]>"
	echo -e "\t<thread sanitizer [ON/OFF]>"
	echo 
	exit
fi

current_dir=`pwd`
build_dir=$1

# create build directory
mkdir -p ${build_dir}
cd ${build_dir}
rm -rf *

# setup cmake
cmake_params=""
cmake_params=${cmake_params}" -DCMAKE_BUILD_TYPE=$2"
cmake_params=${cmake_params}" -DNANOCV_WITH_CUDA=$3"
cmake_params=${cmake_params}" -DNANOCV_WITH_OPENCL=$4"
cmake_params=${cmake_params}" -DNANOCV_WITH_ASAN=$5"
cmake_params=${cmake_params}" -DNANOCV_WITH_LSAN=$6"
cmake_params=${cmake_params}" -DNANOCV_WITH_TSAN=$7"
cmake_params=${cmake_params}" -G Ninja"

cmake ${cmake_params} ${current_dir}/

# build
ninja

# go back to the current directory
cd ${current_dir}


