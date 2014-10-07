#!/bin/bash

source common.sh

# common parameters
params=""
params=${params}${task_svhn}
params=${params}" --loss classnll --trials 10 --threads ${max_threads}"

# models
conv_max="--model forward-network --model-params "
conv_max=${conv_max}"conv:dims=16,rows=6,cols=6;act-snorm;pool-max;"
conv_max=${conv_max}"conv:dims=32,rows=5,cols=5;act-snorm;pool-max;"
conv_max=${conv_max}"conv:dims=64,rows=3,cols=3;act-snorm;pool-max;"
conv_max=${conv_max}"conv:dims=128,rows=2,cols=2;act-snorm;"

conv_min=${conv_max//pool-max/pool-min}
conv_avg=${conv_max//pool-max/pool-avg}
        
mlp0="--model forward-network --model-params "
mlp1=${mlp0}"linear:dims=128;act-snorm;"
mlp2=${mlp1}"linear:dims=64;act-snorm;"
mlp3=${mlp2}"linear:dims=32;act-snorm;"
mlp4=${mlp3}"linear:dims=16;act-snorm;"
mlp5=${mlp4}"linear:dims=8;act-snorm;"

outlayer="linear:dims=10;"

# train models
# TODO
