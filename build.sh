#!/bin/bash

compiler=$CXX
build_type="Release"
generator="ninja"
install_dir="/usr/local/"
install="OFF"

asan_flag="OFF"
msan_flag="OFF"
tsan_flag="OFF"
test_flag="ON"
libcpp_flag="OFF"

# usage
function usage
{
        echo "Usage: "
        echo -e "\t--build-type         <build type [Release/Debug/...]>                default=${build_type}"
        echo -e "\t--generator          <build system [codelite-][ninja/make]>          default=${generator}"
        echo -e "\t--install-dir        <installation directory>                        default=${install_dir}"
        echo -e "\t--install            <install [ON/OFF] Release only>                 default=${install}"
        echo -e "\t--asan               <address sanitizer [ON/OFF]>                    default=${asan_flag}"
        echo -e "\t--msan               <memory sanitizer [ON/OFF]>                     default=${msan_flag}"
        echo -e "\t--tsan               <thread sanitizer [ON/OFF]>                     default=${tsan_flag}"
        echo -e "\t--compiler           <c++ compiler (g++, clang++)>                   optional"
        echo -e "\t--libc++             <use libc++ instead of default libstdc++>       optional"
        echo
}

# read arguments
while [ "$1" != "" ]
do
        case $1 in
                --build-type)   shift
                                build_type=$1
                                ;;
                --generator)    shift
                                generator=$1
                                ;;
                --install-dir)  shift
                                install_dir=$1
                                ;;
                --install)      shift
                                install=$1
                                ;;
                --asan)         shift
                                asan_flag=$1
                                ;;
                --msan)         shift
                                msan_flag=$1
                                ;;
                --tsan)         shift
                                tsan_flag=$1
                                ;;
                --compiler)     shift
                                compiler=$1
                                ;;
                --libc++)       libcpp_flag="ON"
                                ;;
                -h | --help)    usage
                                exit
                                ;;
                * )             echo "unrecognized option $1"
                                echo
                                usage
                                exit 1
        esac
        shift
done

export CXX=${compiler}

current_dir=`pwd`

# create build directory
build_dir=build-$(echo ${build_type} | tr '[:upper:]' '[:lower:]')
#build_dir=build-${build_type,,}
if [ "${asan_flag}" == "ON" ]
then
        build_dir+=-asan
elif [ "${msan_flag}" == "ON" ]
then
        build_dir+=-msan
elif [ "${tsan_flag}" == "ON" ]
then
        build_dir+=-tsan
fi

mkdir -p ${build_dir}
cd ${build_dir}
rm -rf *

# setup build system
if [ "${generator}" == "ninja" ]
then
        generator="Ninja"
        maker="ninja"
        installer="ninja install"

elif [ "${generator}" == "codelite-ninja" ]
then
        generator="CodeLite - Ninja"
        maker="ninja"
        installer="ninja install"

elif [ "${generator}" == "make" ]
then
        generator="Unix Makefiles"
        maker="make -j"
        installer="make install"

elif [ "${generator}" == "codelite-make" ]
then
        generator="CodeLite - Unix Makefiles"
        maker="make -j"
        installer="make install"

else
        echo "Please use a valid generator!"
        usage
        echo
        exit 1
fi

# setup cmake
cmake \
        -DCMAKE_BUILD_TYPE=${build_type} \
        -DNANOCV_WITH_ASAN=${asan_flag} \
        -DNANOCV_WITH_MSAN=${msan_flag} \
        -DNANOCV_WITH_TSAN=${tsan_flag} \
        -DNANOCV_WITH_TESTS=${test_flag} \
        -DNANOCV_WITH_LIBCPP=${libcpp_flag} \
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


