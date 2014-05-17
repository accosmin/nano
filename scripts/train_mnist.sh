#!/bin/bash

source common.sh

# common parameters
params=""
params=${params}${task_mnist}
params=${params}" --loss classdot --trials 10 --threads 1"

#batch="--trainer batch --trainer-params opt=lbfgs,iters=1024,eps=1e-6"
#stochastic="--trainer stochastic --trainer-params opt=sgd,epoch=64"

# trainers (minibatch configurations to evaluate)
minibatch_none="--trainer minibatch --trainer-params batch=1024,iters=128,eps=1e-6,reg=none"
minibatch_l2nm="--trainer minibatch --trainer-params batch=1024,iters=128,eps=1e-6,reg=l2"
minibatch_vari="--trainer minibatch --trainer-params batch=1024,iters=128,eps=1e-6,reg=var"

batch_none="--trainer batch --trainer-params opt=lbfgs,iters=32,eps=1e-6,reg=none"
batch_l2nm="--trainer batch --trainer-params opt=lbfgs,iters=32,eps=1e-6,reg=l2"
batch_vari="--trainer batch --trainer-params opt=lbfgs,iters=32,eps=1e-6,reg=var"

# models
conv0="--model forward-network --model-params "
conv1=${conv0}"conv:dims=16,rows=8,cols=8;snorm;"
conv2=${conv1}"conv:dims=16,rows=7,cols=7;snorm;"
conv3=${conv2}"conv:dims=16,rows=6,cols=6;snorm;"
conv4=${conv3}"conv:dims=16,rows=5,cols=5;snorm;"
conv5=${conv4}"conv:dims=16,rows=4,cols=4;snorm;"
conv6=${conv5}"conv:dims=16,rows=3,cols=3;snorm;"

mlp0="--model forward-network --model-params "
mlp1=${mlp0}"linear:dims=100;snorm;"
mlp2=${mlp1}"linear:dims=100;snorm;"
mlp3=${mlp2}"linear:dims=100;snorm;"
mlp4=${mlp3}"linear:dims=100;snorm;"
mlp5=${mlp4}"linear:dims=100;snorm;"
mlp6=${mlp5}"linear:dims=100;snorm;"

outlayer=";linear:dims=10;softmax:type=global;"

# train models
fn_train ${dir_exp_mnist} minibatchL2-mlp0 ${params} ${minibatch_l2nm} ${mlp0}${outlayer}
fn_train ${dir_exp_mnist} minibatchL2-mlp1 ${params} ${minibatch_l2nm} ${mlp1}${outlayer}
fn_train ${dir_exp_mnist} minibatchL2-mlp2 ${params} ${minibatch_l2nm} ${mlp2}${outlayer}
