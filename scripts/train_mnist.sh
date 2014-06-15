#!/bin/bash

source common.sh

# common parameters
params=""
params=${params}${task_mnist}
params=${params}" --loss class-ratio --trials 10 --threads 6"

# trainers 
stoch_sg="--trainer stochastic --trainer-params opt=sg,epoch=8"
stoch_sga="--trainer stochastic --trainer-params opt=sga,epoch=256"
stoch_sia="--trainer stochastic --trainer-params opt=sia,epoch=256"

mbatch_lbfgs="--trainer minibatch --trainer-params opt=lbfgs,epoch=2048,batch=1024,iters=16,eps=1e-6"
mbatch_cgd="--trainer minibatch --trainer-params opt=cgd,epoch=2048,batch=1024,iters=16,eps=1e-6"
mbatch_gd="--trainer minibatch --trainer-params opt=gd,epoch=128,batch=1024,iters=8,eps=1e-6"

batch_lbfgs="--trainer batch --trainer-params opt=lbfgs,iters=2048,eps=1e-6"
batch_cgd="--trainer batch --trainer-params opt=cgd,iters=2048,eps=1e-6"
batch_gd="--trainer batch --trainer-params opt=gd,iters=2048,eps=1e-6"

# models
conv1="--model forward-network --model-params "
conv1=${conv1}"conv:dims=16,rows=7,cols=7;snorm;pool-abs;"
conv1=${conv1}"conv:dims=16,rows=4,cols=4;snorm;pool-abs;"
conv1=${conv1}"conv:dims=16,rows=4,cols=4;snorm;"
        
conv2="--model forward-network --model-params "
conv2=${conv2}"conv:dims=16,rows=7,cols=7;snorm;pool-abs;"
conv2=${conv2}"conv:dims=16,rows=6,cols=6;snorm;pool-abs;"
conv2=${conv3}"conv:dims=16,rows=3,cols=3;snorm;"
       
conv3="--model forward-network --model-params "
conv3=${conv3}"conv:dims=16,rows=5,cols=5;snorm;pool-abs;"
conv3=${conv3}"conv:dims=16,rows=5,cols=5;snorm;pool-abs;"
conv3=${conv3}"conv:dims=16,rows=4,cols=4;snorm;"

mlp0="--model forward-network --model-params "
mlp1=${mlp0}"linear:dims=64;snorm;"
mlp2=${mlp1}"linear:dims=64;snorm;"
mlp3=${mlp2}"linear:dims=64;snorm;"
mlp4=${mlp3}"linear:dims=64;snorm;"
mlp5=${mlp4}"linear:dims=64;snorm;"
mlp6=${mlp5}"linear:dims=64;snorm;"

outlayer=";linear:dims=10;softmax:type=global;"

# train models
#for trainer in `echo "stoch_sg stoch_sga stoch_sia mbatch_gd mbatch_cgd mbatch_lbfgs batch_gd batch_cgd batch_lbfgs"`
for trainer in `echo "mbatch_gd"`
do
        fn_train ${dir_exp_mnist} mlp0_${trainer} ${params} ${!trainer} ${mlp0}${outlayer}
        fn_train ${dir_exp_mnist} mlp1_${trainer} ${params} ${!trainer} ${mlp1}${outlayer}
        #fn_train ${dir_exp_mnist} mlp2_${trainer} ${params} ${!trainer} ${mlp2}${outlayer}
        #fn_train ${dir_exp_mnist} mlp3_${trainer} ${params} ${!trainer} ${mlp3}${outlayer}
        #fn_train ${dir_exp_mnist} conv1_${trainer} ${params} ${!trainer} ${conv1}${outlayer}
        #fn_train ${dir_exp_mnist} conv2_${trainer} ${params} ${!trainer} ${conv2}${outlayer}
        #fn_train ${dir_exp_mnist} conv3_${trainer} ${params} ${!trainer} ${conv3}${outlayer}
done
