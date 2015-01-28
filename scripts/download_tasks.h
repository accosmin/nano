#!/bin/bash

source common_train.sh

# SVHN dataset
mkdir ${dir_db_svhn}

wget http://ufldl.stanford.edu/housenumbers/train_32x32.mat -P ${dir_db_svhn}
wget http://ufldl.stanford.edu/housenumbers/extra_32x32.mat -P ${dir_db_svhn}
wget http://ufldl.stanford.edu/housenumbers/test_32x32.mat -P ${dir_db_svhn}

# MNIST dataset
mkdir ${dir_db_mnist}

wget http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz -P ${dir_db_mnist}
wget http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz -P ${dir_db_mnist}
wget http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz -P ${dir_db_mnist}
wget http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz -P ${dir_db_mnist}

# STL10 dataset
mkdir ${dir_db_stl10}

wget http://ai.stanford.edu/~acoates/stl10/stl10_matlab.tar.gz -P ${dir_db_stl10}

# CIFAR10 dataset
mkdir ${dir_db_cifar10}

wget http://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz -P ${dir_db_cifar10} 

# CIFAR100 dataset
mkdir ${dir_db_cifar100}

wget http://www.cs.toronto.edu/~kriz/cifar-100-binary.tar.gz -P ${dir_db_cifar100}

# NORB dataset
mkdir ${dir_db_norb}

# todo
