#!/bin/bash

source train_common.sh

# paths
dir_results=/home/cosmin/experiments/results
dir_db=/home/cosmin/experiments/databases
trainer=./build-release/ncv_trainer

dir_exp=${dir_results}/mnist
mkdir -p ${dir_exp}

# common parameters
batch="opt=lbfgs,iters=64,eps=1e-6"
minibatch="batch=1024,iters=8000,eps=1e-6"
stoch="opt=sgd,epoch=64"

param=""
param=${param}"--task mnist --task-dir ${dir_db}/mnist/ "
param=${param}"--loss logistic --trials 1 --threads 4"

# models
model0=""

model1=${model0}"conv:dims=32,rows=7,cols=7;snorm;smax-abs-pool;"
model1=${model1}"conv:dims=32,rows=4,cols=4;snorm;smax-abs-pool;"
model1=${model1}"conv:dims=32,rows=4,cols=4;snorm;"

model2=${model0}"conv:dims=32,rows=7,cols=7;snorm;smax-abs-pool;"
model2=${model2}"conv:dims=32,rows=6,cols=6;snorm;smax-abs-pool;"
model2=${model2}"conv:dims=32,rows=3,cols=3;snorm;"

model3=${model0}"conv:dims=32,rows=5,cols=5;snorm;smax-abs-pool;"
model3=${model3}"conv:dims=32,rows=5,cols=5;snorm;smax-abs-pool;"
model3=${model3}"conv:dims=32,rows=4,cols=4;snorm;"

# train models
#fn_train forward-network ${model0} stochastic ${stoch} model0
#fn_train forward-network ${model1} stochastic ${stoch} model1
#fn_train forward-network ${model2} stochastic ${stoch} model2
#fn_train forward-network ${model3} stochastic ${stoch} model3

#fn_train forward-network ${model0} batch ${batch} model0
#fn_train forward-network ${model1} batch ${batch} model1
#fn_train forward-network ${model2} batch ${batch} model2
#fn_train forward-network ${model3} batch ${batch} model3

fn_train forward-network ${model0} minibatch ${minibatch} model0
fn_train forward-network ${model1} minibatch ${minibatch} model1
fn_train forward-network ${model2} minibatch ${minibatch} model2
fn_train forward-network ${model3} minibatch ${minibatch} model3

