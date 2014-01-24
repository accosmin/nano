#!/bin/bash

source train_common.sh

# paths
dir_results=/home/cosmin/experiments/results
dir_db=/home/cosmin/experiments/databases
trainer=./build/ncv_trainer

dir_exp=${dir_results}/mnist
mkdir -p ${dir_exp}

# common parameters
batch="opt=lbfgs,eps=1e-6,iters=1024"
stoch="opt=sgd,alpha=1e-2,epoch=64"

param=""
param=${param}"--task mnist --task-dir ${dir_db}/mnist/ "
param=${param}"--loss classnll --trials 10 --threads 4"

# models
network0=""
network1=${network0}"conv:count=16,rows=9,cols=9;snorm;"
network2=${network1}"conv:count=16,rows=9,cols=9;snorm;"
network3=${network2}"conv:count=16,rows=7,cols=7;snorm;"
network4=${network3}"conv:count=16,rows=6,cols=6;snorm;"

#valgrind --tool=memcheck --leak-check=yes ./build/ncv_trainer ${params}

# train models
fn_train forward-network ${network0} stochastic ${stoch} model0
fn_train forward-network ${network0} batch ${batch} model0

#fn_train forward-network ${network4} stochastic ${stoch} model4
#fn_train forward-network ${network4} batch ${batch} model4

