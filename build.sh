#!/bin/bash

compiler=$CXX
build_dir=""
build_type="release"
generator="make"
install_dir="/usr/local/"
install="OFF"

asan_flag="OFF"
msan_flag="OFF"
tsan_flag="OFF"
test_flag="ON"
libcpp_flag="OFF"
gold_flag="OFF"
lto_flag="OFF"
werror_flag="OFF"
float_flag="ON"
double_flag="OFF"
long_double_flag="OFF"

# usage
function usage
{
        echo "usage: "
        echo -e "\t--build-type         <build type [release|debug|relwithdebinfo]>     default=${build_type}"
        echo -e "\t--build-dir          <build directory>                               defaut=${build_dir}"
        echo -e "\t--generator          <build system [codelite-][ninja|make]>          default=${generator}"
        echo -e "\t--install-dir        <installation directory>                        default=${install_dir}"
        echo -e "\t--install            <install [ON/OFF] Release only>                 default=${install}"
        echo -e "\t--asan               <with address sanitizer>                        optional"
        echo -e "\t--msan               <with memory sanitizer>                         optional"
        echo -e "\t--tsan               <with thread sanitizer>                         optional"
        echo -e "\t--compiler           <c++ compiler [g++|clang++|...]>                optional"
        echo -e "\t--libc++             <use libc++ instead of default libstdc++>       optional"
        echo -e "\t--gold               <use gold linker instead of default linker>     optional"
        echo -e "\t--lto                <use link time optimization>                    optional"
        echo -e "\t--werror             <stop compilation at first warning>             optional"
        echo -e "\t--float              <use float as the default scalar>               optional"
        echo -e "\t--double             <use double as the default scalar>              optional"
        echo -e "\t--long-double        <use long double as the default scalar>         optional"
        echo
}

# read arguments
while [ "$1" != "" ]
do
        case $1 in
                --build-type)   shift
                                build_type=$1
                                ;;
                --build-dir)    shift
                                build_dir=$1
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
                --asan)         asan_flag="ON"
                                ;;
                --msan)         msan_flag="ON"
                                ;;
                --tsan)         tsan_flag="ON"
                                ;;
                --compiler)     shift
                                compiler=$1
                                ;;
                --libc++)       libcpp_flag="ON"
                                ;;
                --gold)         gold_flag="ON"
                                ;;
                --lto)          lto_flag="ON"
                                ;;
                --werror)       werror_flag="ON"
                                ;;
                --float)        float_flag="ON"
                                double_flag="OFF"
                                long_double_flag="OFF"
                                ;;
                --double)       float_flag="OFF"
                                double_flag="ON"
                                long_double_flag="OFF"
                                ;;
                --long-double)  float_flag="OFF"
                                double_flag="OFF"
                                long_double_flag="ON"
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
if [[ -z "${build_dir}" ]]
then
        build_dir=build-$(echo ${build_type} | tr '[:upper:]' '[:lower:]')
        #build_dir=build-${build_type,,}
        if [ "${lto_flag}" == "ON" ]
        then
                build_dir+=-lto
        elif [ "${asan_flag}" == "ON" ]
        then
                build_dir+=-asan
        elif [ "${msan_flag}" == "ON" ]
        then
                build_dir+=-msan
        elif [ "${tsan_flag}" == "ON" ]
        then
                build_dir+=-tsan
        fi
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
        maker="make -j6"
        installer="make install"

elif [ "${generator}" == "codelite-make" ]
then
        generator="CodeLite - Unix Makefiles"
        maker="make -j6"
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
        -DNANO_WITH_ASAN=${asan_flag} \
        -DNANO_WITH_MSAN=${msan_flag} \
        -DNANO_WITH_TSAN=${tsan_flag} \
        -DNANO_WITH_TESTS=${test_flag} \
        -DNANO_WITH_LIBCPP=${libcpp_flag} \
        -DNANO_WITH_GOLD=${gold_flag} \
        -DNANO_WITH_LTO=${lto_flag} \
        -DNANO_WITH_WERROR=${werror_flag} \
        -DNANO_WITH_FLOAT_SCALAR=${float_flag} \
        -DNANO_WITH_DOUBLE_SCALAR=${double_flag} \
        -DNANO_WITH_LONG_DOUBLE_SCALAR=${long_double_flag} \
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
