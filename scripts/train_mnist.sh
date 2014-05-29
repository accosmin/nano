#!/bin/bash

source common.sh

# common parameters
params=""
params=${params}${task_mnist}
params=${params}" --loss class-ratio --trials 1 --threads 1"

# trainers 
stoch_sg="--trainer stochastic --trainer-params opt=sg,epoch=16"
stoch_sga="--trainer stochastic --trainer-params opt=sga,epoch=16"
stoch_sia="--trainer stochastic --trainer-params opt=sia,epoch=16"

mbatch_lbfgs="--trainer minibatch --trainer-params opt=lbfgs,epochs=64,batch=1024,iters=8,eps=1e-6"
mbatch_cgd="--trainer minibatch --trainer-params opt=cgd,epochs=64,batch=1024,iters=8,eps=1e-6"
mbatch_gd="--trainer minibatch --trainer-params opt=gd,epochs=64,batch=1024,iters=8,eps=1e-6"

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
for trainer in `echo "stoch_sg stoch_sga stoch_sia mbatch_gd mbatch_cgd mbatch_lbfgs batch_gd batch_cgd batch_lbfgs"`
do
        fn_train ${dir_exp_mnist} mlp0_${trainer} ${params} ${!trainer} ${mlp0}${outlayer}
done
