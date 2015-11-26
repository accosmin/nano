#!/bin/bash

compiler=$CXX
generator=ninja

build_dir="`pwd`/build-release"
build_type=Release
install_dir="`pwd`/install"

if [ -z "${compiler}" ]
then
        compiler=g++
fi

# usage
function usage
{
	echo "Usage: "
	echo -e "\t--generator          <build system [codelite-][ninja/make]>	default=${generator}"
	echo -e "\t--compiler           <c++ compiler (g++, clang++)>		optional"
	echo
}

# read arguments
while [ "$1" != "" ]
do
	case $1 in
                --generator)	shift
                                generator=$1
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

bash build.sh \
	--compiler ${compiler} \
	--build-dir ${build_dir} \
	--build-type ${build_type} \
	--generator ${generator} \
	--install-dir ${install_dir} \
	--install OFF \
	--asan OFF --tsan OFF


