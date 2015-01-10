#!/bin/bash

build_dir=""
build_type="Release"

cuda_flag="OFF"
opencl_flag="OFF"

asan_flag="OFF"
lsan_flag="OFF"
tsan_flag="OFF"

# usage
function usage
{
	echo "Usage: "
	echo -e "\t--build-dir          <build directory>" 
	echo -e "\t--build-type         <build type [Release/Debug]>    default=${build_type}"
	echo -e "\t--cuda               <CUDA flag [ON/OFF]>            default=${cuda_flag}"
	echo -e "\t--opencl             <OpenCL flag [ON/OFF]>          default=${opencl_flag}"
	echo -e "\t--asan               <address sanitizer [ON/OFF]>    default=${asan_flag}"
	echo -e "\t--lsan               <leak sanitizer [ON/OFF]>       default=${lsan_flag}"
	echo -e "\t--tsan               <thread sanitizer [ON/OFF]>     default=${tsan_flag}"
	echo
}

# read arguments
while [ "$1" != "" ]
do
	case $1 in
        	--build-dir)	shift
                                build_dir=$1
                                ;;
        	--build-type)	shift
                                build_type=$1
                                ;;
        	--cuda)		shift
                                cuda_flag=$1
                                ;;
        	--opencl)	shift
                                opencl_flag=$1
                                ;;
        	--asan)		shift
                                asan_flag=$1
                                ;;
        	--lsan)		shift
                                lsan_flag=$1
                                ;;
        	--tsan)		shift
                                tsan_flag=$1
                                ;;
		-h | --help)	usage
				exit
				;;
		* )		echo "unrecognized option $1"
				echo
				usage
                                exit 1
	esac
	shift
done

current_dir=`pwd`

# create build directory
if [ -z "${build_dir}" ]
then
	echo "please provide a build directory!"
	echo
	exit 1
fi

mkdir -p ${build_dir}
cd ${build_dir}
rm -rf *

# setup cmake
cmake_params=""
cmake_params=${cmake_params}" -DCMAKE_BUILD_TYPE=${build_type}"
cmake_params=${cmake_params}" -DNANOCV_WITH_CUDA=${cuda_flag}"
cmake_params=${cmake_params}" -DNANOCV_WITH_OPENCL=${opencl_flag}"
cmake_params=${cmake_params}" -DNANOCV_WITH_ASAN=${asan_flag}"
cmake_params=${cmake_params}" -DNANOCV_WITH_LSAN=${lsan_flag}"
cmake_params=${cmake_params}" -DNANOCV_WITH_TSAN=${tsan_flag}"
cmake_params=${cmake_params}" -G Ninja"

cmake ${cmake_params} ${current_dir}/

# build
ninja
echo

# go back to the current directory
cd ${current_dir}


