#!/bin/bash

source common_train.sh

# SVHN dataset
mkdir -p ${dir_db_svhn}

wget -N http://ufldl.stanford.edu/housenumbers/train_32x32.mat -P ${dir_db_svhn}
wget -N http://ufldl.stanford.edu/housenumbers/extra_32x32.mat -P ${dir_db_svhn}
wget -N http://ufldl.stanford.edu/housenumbers/test_32x32.mat -P ${dir_db_svhn}

# MNIST dataset
mkdir -p ${dir_db_mnist}

wget -N http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz -P ${dir_db_mnist}
wget -N http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz -P ${dir_db_mnist}
wget -N http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz -P ${dir_db_mnist}
wget -N http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz -P ${dir_db_mnist}

# STL10 dataset
mkdir -p ${dir_db_stl10}

wget -N http://ai.stanford.edu/~acoates/stl10/stl10_binary.tar.gz -P ${dir_db_stl10}

# CIFAR10 dataset
mkdir -p ${dir_db_cifar10}

wget -N http://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz -P ${dir_db_cifar10}

# CIFAR100 dataset
mkdir -p ${dir_db_cifar100}

wget -N http://www.cs.toronto.edu/~kriz/cifar-100-binary.tar.gz -P ${dir_db_cifar100}
