#!/bin/bash

compiler=$CXX
build_dir=""
build_type="Release"
build_sys="ninja"
install_dir="/usr/local/"
install="OFF"

asan_flag="OFF"
tsan_flag="OFF"

if [ -z "${compiler}" ]
then
        compiler=g++
fi

# usage
function usage
{
	echo "Usage: "
	echo -e "\t--build-dir          <build directory>               required" 
	echo -e "\t--build-type         <build type [Release/Debug]>    default=${build_type}"
	echo -e "\t--build-sys          <build system [ninja/make]>	default=${build_sys}"
	echo -e "\t--install-dir        <installation directory>        default=${install_dir}" 
	echo -e "\t--install            <install [ON/OFF] Release only> default=${install}" 
	echo -e "\t--asan               <address sanitizer [ON/OFF]>    default=${asan_flag}"
	echo -e "\t--tsan               <thread sanitizer [ON/OFF]>     default=${tsan_flag}"
	echo -e "\t--compiler           <c++ compiler (g++, clang++)>	optional"
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
        	--build-sys)	shift
                                build_sys=$1
                                ;;
        	--install-dir)	shift
                                install_dir=$1
                                ;;
        	--install)	shift
                                install=$1
                                ;;
        	--asan)		shift
                                asan_flag=$1
                                ;;
        	--tsan)		shift
                                tsan_flag=$1
                                ;;
        	--compiler)	shift
			        compiler=$1
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

export CXX=${compiler}

current_dir=`pwd`

# create build directory
if [ -z "${build_dir}" ]
then
	echo "Please provide a build directory!"
	echo
	exit 1
fi

mkdir -p ${build_dir}
cd ${build_dir}
rm -rf *

# setup build systemr
if [ "${build_sys}" == "ninja" ]
then
	generator="Ninja"
	maker="ninja"
	installer="ninja install"

elif [ "${build_sys}" == "make" ]
then
	generator="Unix Makefiles"
	maker="make -j"
	installer="make install"

else
	echo "Please use either ninja or make as the build system!"
	echo
	exit 1
fi

# setup cmake
cmake \
	-DCMAKE_BUILD_TYPE=${build_type} \
    	-DNANOCV_WITH_ASAN=${asan_flag} \
    	-DNANOCV_WITH_TSAN=${tsan_flag} \
    	-G "${generator}" \
    	-DCMAKE_INSTALL_PREFIX=${install_dir} \
    	${current_dir}/

# build
${maker}
echo

# install
if [ "Release" == "${build_type}" ] && [ "ON" == "${install}" ]
then
	${installer}
	echo
fi

# go back to the current directory
cd ${current_dir}


