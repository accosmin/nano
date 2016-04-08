#!/bin/bash

compiler=$CXX
generator=ninja
libcpp=""
gold=""

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
        echo -e "\t--libc++             <use libc++ instead of default libstdc++>       optional"
        echo -e "\t--gold               <use gold linker instead of default linker>     optional"
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
                --gold)         gold="--gold"
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
        --asan OFF --msan OFF --tsan OFF ${libcpp} ${gold}

bash build.sh \
        --compiler ${compiler} \
        --build-type RelWithDebInfo \
        --generator ${generator} \
        --install-dir ${install_dir} \
        --install OFF \
        --asan OFF --msan OFF --tsan OFF ${libcpp} ${gold}

