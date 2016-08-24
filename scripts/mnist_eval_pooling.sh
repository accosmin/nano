#!/bin/bash

source $(dirname $0)/common_train.sh

fn_cmdline $*

# common parameters
common="${task_mnist} ${loss_classnll}"
outdir="${dir_exp_mnist}/eval_pooling"
mkdir -p ${outdir}

# models
conv1="--model forward-network --model-params "
conv1=${conv1}"conv:dims=64,rows=9,cols=9,conn=1;act-snorm;pool-full;"
conv1=${conv1}"conv:dims=64,rows=5,cols=5,conn=8;act-snorm;pool-full;"
conv1=${conv1}"conv:dims=64,rows=3,cols=3,conn=8;act-snorm;"

conv1_full=${conv1//pool-full/pool-full}
conv1_soft=${conv1//pool-full/pool-soft}
conv1_gauss=${conv1//pool-full/pool-gauss}

conv2="--model forward-network --model-params "
conv2=${conv2}"conv:dims=64,rows=5,cols=5,conn=1;act-snorm;pool-full;"
conv2=${conv2}"conv:dims=64,rows=5,cols=5,conn=8;act-snorm;pool-full;"
conv2=${conv2}"conv:dims=64,rows=3,cols=3,conn=8;act-snorm;"

conv2_full=${conv2//pool-full/pool-full}
conv2_soft=${conv2//pool-full/pool-soft}
conv2_gauss=${conv2//pool-full/pool-gauss}

outlayer="affine:dims=10;"

models=${models}" conv1_full conv1_soft conv1_gauss"
models=${models}" conv2_full conv2_soft conv2_gauss"

# trainers
fn_make_trainers "stop_early"
trainers="stoch_adam"

# criteria
criteria="crit_avg"

# train models
fn_train "${models}" "${trainers}" "${criteria}"

# compare models
for ((trial=0;trial<${trials};trial++))
do
        bash $(dirname $0)/plot_models.sh ${outdir}/trial${trial}_conv1.pdf ${outdir}/trial${trial}_*conv1*.state
        bash $(dirname $0)/plot_models.sh ${outdir}/trial${trial}_conv2.pdf ${outdir}/trial${trial}_*conv2*.state
done

fn_sumarize "${outdir}" "${models}" "${trainers}" "${criteria}"
