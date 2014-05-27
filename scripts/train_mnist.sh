#!/bin/bash

source common.sh

# common parameters
params=""
params=${params}${task_mnist}
params=${params}" --loss class-ratio --trials 1 --threads 1"

# trainers 
stoch_sg="--trainer stochastic --trainer-params opt=sg,epoch=64"
stoch_sga="--trainer stochastic --trainer-params opt=sga,epoch=64"
stoch_sia="--trainer stochastic --trainer-params opt=sia,epoch=64"

minibatch_lbfgs="--trainer minibatch --trainer-params opt=lbfgs,epochs=64,batch=1024,iters=8,eps=1e-6"
minibatch_cgd="--trainer minibatch --trainer-params opt=cgd,epochs=64,batch=1024,iters=8,eps=1e-6"
minibatch_gd="--trainer minibatch --trainer-params opt=gd,epochs=64,batch=1024,iters=8,eps=1e-6"

batch_lbfgs="--trainer batch --trainer-params opt=lbfgs,iters=256,eps=1e-6"
batch_cgd="--trainer batch --trainer-params opt=cgd,iters=256,eps=1e-6"
batch_gd="--trainer batch --trainer-params opt=gd,iters=256,eps=1e-6"

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
fn_train ${dir_exp_mnist} stoch-sg-mlp0 ${params} ${stoch_sg} ${mlp0}${outlayer}
fn_train ${dir_exp_mnist} stoch-sg-mlp1 ${params} ${stoch_sg} ${mlp1}${outlayer}
fn_train ${dir_exp_mnist} stoch-sg-mlp2 ${params} ${stoch_sg} ${mlp2}${outlayer}
fn_train ${dir_exp_mnist} stoch-sg-mlp3 ${params} ${stoch_sg} ${mlp3}${outlayer}

fn_train ${dir_exp_mnist} stoch-sga-mlp0 ${params} ${stoch_sga} ${mlp0}${outlayer}
fn_train ${dir_exp_mnist} stoch-sga-mlp1 ${params} ${stoch_sga} ${mlp1}${outlayer}
fn_train ${dir_exp_mnist} stoch-sga-mlp2 ${params} ${stoch_sga} ${mlp2}${outlayer}
fn_train ${dir_exp_mnist} stoch-sga-mlp3 ${params} ${stoch_sga} ${mlp3}${outlayer}

fn_train ${dir_exp_mnist} stoch-sia-mlp0 ${params} ${stoch_sia} ${mlp0}${outlayer}
fn_train ${dir_exp_mnist} stoch-sia-mlp1 ${params} ${stoch_sia} ${mlp1}${outlayer}
fn_train ${dir_exp_mnist} stoch-sia-mlp2 ${params} ${stoch_sia} ${mlp2}${outlayer}
fn_train ${dir_exp_mnist} stoch-sia-mlp3 ${params} ${stoch_sia} ${mlp3}${outlayer}

