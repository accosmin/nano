#!/bin/bash

source train_common.sh

# paths
dir_exp=${dir_results}/cifar10
mkdir -p ${dir_exp}

# common parameters
param=""
param=${param}"--task cifar10 --task-dir ${dir_db}/cifar10/ "
param=${param}"--loss logistic-sum --trials 1 --threads 4"

# models
conv0=""

conv1=${conv0}"conv:dims=32,rows=7,cols=7;snorm;smax-abs-pool;"
conv1=${conv1}"conv:dims=32,rows=6,cols=6;snorm;smax-abs-pool;"
conv1=${conv1}"conv:dims=32,rows=4,cols=4;snorm;"

conv2=${conv0}"conv:dims=32,rows=9,cols=9;snorm;smax-abs-pool;"
conv2=${conv2}"conv:dims=32,rows=7,cols=7;snorm;smax-abs-pool;"
conv2=${conv2}"conv:dims=32,rows=3,cols=3;snorm;"

conv3=${conv0}"conv:dims=32,rows=9,cols=9;snorm;smax-abs-pool;"
conv3=${conv3}"conv:dims=32,rows=5,cols=5;snorm;smax-abs-pool;"
conv3=${conv3}"conv:dims=32,rows=4,cols=4;snorm;"

mlp0=""
mlp1=${mlp0}"linear:dims=100;snorm;"
mlp2=${mlp1}"linear:dims=100;snorm;"
mlp3=${mlp2}"linear:dims=100;snorm;"
mlp4=${mlp3}"linear:dims=100;snorm;"

# train models
#fn_train forward-network ${conv0} stochastic ${stoch} conv0
#fn_train forward-network ${conv1} stochastic ${stoch} conv1
#fn_train forward-network ${conv2} stochastic ${stoch} conv2
#fn_train forward-network ${conv3} stochastic ${stoch} conv3

#fn_train forward-network ${conv0} batch ${batch} conv0
#fn_train forward-network ${conv1} batch ${batch} conv1
#fn_train forward-network ${conv2} batch ${batch} conv2
#fn_train forward-network ${conv3} batch ${batch} conv3

#fn_train forward-network ${conv0} minibatch ${minibatch} conv0
#fn_train forward-network ${conv1} minibatch ${minibatch} conv1
#fn_train forward-network ${conv2} minibatch ${minibatch} conv2
#fn_train forward-network ${conv3} minibatch ${minibatch} conv3

fn_train forward-network ${mlp0} minibatch ${minibatch} mlp0
fn_train forward-network ${mlp1} minibatch ${minibatch} mlp1
fn_train forward-network ${mlp2} minibatch ${minibatch} mlp2
fn_train forward-network ${mlp3} minibatch ${minibatch} mlp3
fn_train forward-network ${mlp4} minibatch ${minibatch} mlp4
