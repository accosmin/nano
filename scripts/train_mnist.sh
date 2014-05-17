#!/bin/bash

source common.sh

# common parameters
params=""
params=${params}${task_mnist}
params=${params}" --loss class-ratio --trials 10 --threads 1"

# trainers 
stochastic="--trainer stochastic --trainer-params opt=sgd,epoch=256"
minibatch="--trainer minibatch --trainer-params batch=1024,iters=256,eps=1e-6"
batch="--trainer batch --trainer-params opt=cgd,iters=256,eps=1e-6,reg=none"

# models
conv0="--model forward-network --model-params "
conv1=${conv0}"conv:dims=16,rows=8,cols=8;snorm;"
conv2=${conv1}"conv:dims=16,rows=7,cols=7;snorm;"
conv3=${conv2}"conv:dims=16,rows=6,cols=6;snorm;"
conv4=${conv3}"conv:dims=16,rows=5,cols=5;snorm;"
conv5=${conv4}"conv:dims=16,rows=4,cols=4;snorm;"
conv6=${conv5}"conv:dims=16,rows=3,cols=3;snorm;"

mlp0="--model forward-network --model-params "
mlp1=${mlp0}"linear:dims=64;snorm;"
mlp2=${mlp1}"linear:dims=64;snorm;"
mlp3=${mlp2}"linear:dims=64;snorm;"
mlp4=${mlp3}"linear:dims=64;snorm;"
mlp5=${mlp4}"linear:dims=64;snorm;"
mlp6=${mlp5}"linear:dims=64;snorm;"

outlayer=";linear:dims=10;softmax:type=global;"

# train models
fn_train ${dir_exp_mnist} stochastic-mlp0 ${params} ${stochastic} ${mlp0}${outlayer}
fn_train ${dir_exp_mnist} stochastic-mlp1 ${params} ${stochastic} ${mlp1}${outlayer}
fn_train ${dir_exp_mnist} stochastic-mlp2 ${params} ${stochastic} ${mlp2}${outlayer}

fn_train ${dir_exp_mnist} minibatch-mlp0 ${params} ${minibatch} ${mlp0}${outlayer}
fn_train ${dir_exp_mnist} minibatch-mlp1 ${params} ${minibatch} ${mlp1}${outlayer}
fn_train ${dir_exp_mnist} minibatch-mlp2 ${params} ${minibatch} ${mlp2}${outlayer}

fn_train ${dir_exp_mnist} batch-mlp0 ${params} ${batch} ${mlp0}${outlayer}
fn_train ${dir_exp_mnist} batch-mlp1 ${params} ${batch} ${mlp1}${outlayer}
fn_train ${dir_exp_mnist} batch-mlp2 ${params} ${batch} ${mlp2}${outlayer}
