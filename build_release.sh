#!/bin/bash

compiler=$CXX
generator=ninja
libcpp=""
gold=""
lto=""
float=""
double=""
long_double=""
opencl=""

install_dir="`pwd`/install"

if [ -z "${compiler}" ]
then
        compiler=g++
fi

# usage
function usage
{
        echo "Usage: "
        echo -e "\t--generator          <build system [codelite-][ninja/make]>          default=${generator}"
        echo -e "\t--compiler           <c++ compiler (g++, clang++)>                   optional"
        echo -e "\t--opencl             <use OpenCL>                                    optional"
        echo -e "\t--libc++             <use libc++ instead of default libstdc++>       optional"
        echo -e "\t--gold               <use gold linker instead of default linker>     optional"
        echo -e "\t--lto                <use link time optimization>                    optional"
        echo -e "\t--float              <use float as the default scalar>               optional"
        echo -e "\t--double             <use double as the default scalar>              optional"
        echo -e "\t--long-double        <use long double as the default scalar>         optional"
        echo
}

# read arguments
while [ "$1" != "" ]
do
        case $1 in
                --generator)    shift
                                generator=$1
                                ;;
                --compiler)     shift
                                compiler=$1
                                ;;
                --libc++)       libcpp="--libc++"
                                ;;
                --opencl)       opencl="--opencl"
                                ;;
                --gold)         gold="--gold"
                                ;;
                --lto)          lto="--lto"
                                ;;
                --float)        float="--float"
                                ;;
                --double)       double="--double"
                                ;;
                --long-double)  long_double="--long-double"
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

bash build.sh \
        --compiler ${compiler} \
        --build-type Release \
        --generator ${generator} \
        --install-dir ${install_dir} \
        --install OFF \
        --asan OFF --msan OFF --tsan OFF ${libcpp} ${gold} ${lto} ${float} ${double} ${long_double} ${opencl}

bash build.sh \
        --compiler ${compiler} \
        --build-type RelWithDebInfo \
        --generator ${generator} \
        --install-dir ${install_dir} \
        --install OFF \
        --asan OFF --msan OFF --tsan OFF ${libcpp} ${gold} ${float} ${double} ${long_double} ${opencl}

