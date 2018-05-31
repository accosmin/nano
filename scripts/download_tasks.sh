#!/bin/bash

dir_exp=$HOME/experiments/results
dir_db=$HOME/experiments/databases

dir_db_iris=${dir_db}/iris/
dir_db_wine=${dir_db}/wine/
dir_db_mnist=${dir_db}/mnist/
dir_db_cifar10=${dir_db}/cifar10/
dir_db_cifar100=${dir_db}/cifar100/
dir_db_fashion_mnist=${dir_db}/fashion-mnist/

# SVHN dataset
function download_svhn {
	local dir=${dir_db}/svhn/
	mkdir -p ${dir}

	wget -N http://ufldl.stanford.edu/housenumbers/train_32x32.mat -P ${dir}
	wget -N http://ufldl.stanford.edu/housenumbers/extra_32x32.mat -P ${dir}
	wget -N http://ufldl.stanford.edu/housenumbers/test_32x32.mat -P ${dir}
}

# MNIST dataset
function download_mnist {
	local dir=${dir_db}/mnist
	mkdir -p ${dir}

	wget -N http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz -P ${dir}
	wget -N http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz -P ${dir}
	wget -N http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz -P ${dir}
	wget -N http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz -P ${dir}
}

# Fashion-MNIST dataset
function download_fashion_mnist {
	local dir=${dir_db}/fashion-mnist
	mkdir -p ${dir}

	wget -N http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz -P ${dir}
	wget -N http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz -P ${dir}
	wget -N http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz -P ${dir}
	wget -N http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz -P ${dir}
}

# CIFAR10 dataset
function download_cifar10 {
	local dir=${dir_db}/cifar10
	mkdir -p ${dir}

	wget -N http://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz -P ${dir}
}

# CIFAR100 dataset
function download_cifar100 {
	local dir=${dir_db}/cifar100
	mkdir -p ${dir}

	wget -N http://www.cs.toronto.edu/~kriz/cifar-100-binary.tar.gz -P ${dir}
}

# IRIS dataset
function download_iris {
	local dir=${dir_db}/iris
	mkdir -p ${dir}

	wget -N http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data -P ${dir}
}

# WINE dataset
function download_wine {
	local dir=${dir_db}/wine
	mkdir -p ${dir}

	wget -N http://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data -P ${dir}
}

# IRIS dataset
function download_iris {
	local dir=${dir_db}/iris
	mkdir -p ${dir}

	wget -N http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data -P ${dir}
}

# Process command line
function usage {
	cat <<EOF
usage: $0 [OPTIONS]

options:
	-h,--help		
		print usage
	--mnist
		download MNIST task
	--fashion-minst
		download Fashion-MNIST task
	--iris
		download IRIS task
	--wine	
		download WINE task
	--svhn
		download SVHN task
	--cifar10
		download CIFAR-10 task
	--cifar100
		donwload CIFAR-100 task
EOF
	exit 1
}

if [ "$1" == "" ]; then
	usage
fi

while [ "$1" != "" ]; do
	case $1 in 
		-h | --help)		usage
					;;
		--wine)			download_wine
					;;
		--iris)			download_iris
					;;
		--mnist)		download_mnist
					;;
		--fashion-mnist)	download_fashion_mnist
					;;	
		--svhn)			download_svhn
					;;
		--cifar10)		download_cifar10
					;;
		--cifar100)		download_cifar100
					;;
		*)			echo "unrecognized option $1"
					echo
					usage
					;;
	esac
	shift
done

exit 0
