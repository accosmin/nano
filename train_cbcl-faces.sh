#!/bin/bash

source train_common.sh

# paths
dir_results=/home/cosmin/experiments/results
dir_db=/home/cosmin/experiments/databases
trainer=./build-release/ncv_trainer

dir_exp=${dir_results}/cbcl-faces
mkdir -p ${dir_exp}

# common parameters
batch="opt=lbfgs,iters=1024,eps=1e-6"
minibatch="batch=1024,iters=1024,eps=1e-6"
stoch="opt=sgd,epoch=64"

param=""
param=${param}"--task cbcl-faces --task-dir ${dir_db}/cbcl-faces/ "
param=${param}"--loss logistic --trials 1 --threads 4"

# models
model0=""

model1=${model0}"conv:dims=32,rows=8,cols=8;snorm;smax-abs-pool;"
model1=${model1}"conv:dims=32,rows=6,cols=6;snorm;"

model2=${model0}"conv:dims=32,rows=6,cols=6;snorm;smax-abs-pool;"
model2=${model2}"conv:dims=32,rows=4,cols=4;snorm;smax-abs-pool;"
model2=${model2}"conv:dims=32,rows=2,cols=2;snorm;"

# train models
#fn_train forward-network ${model0} stochastic ${stoch} model0
#fn_train forward-network ${model1} stochastic ${stoch} model1
#fn_train forward-network ${model2} stochastic ${stoch} model2

fn_train forward-network ${model0} batch ${batch} model0
fn_train forward-network ${model1} batch ${batch} model1
#fn_train forward-network ${model2} batch ${batch} model2

#fn_train forward-network ${model0} minibatch ${minibatch} model0
#fn_train forward-network ${model1} minibatch ${minibatch} model1
#fn_train forward-network ${model2} minibatch ${minibatch} model2
