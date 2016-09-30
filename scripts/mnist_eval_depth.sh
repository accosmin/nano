#!/bin/bash

source $(dirname $0)/common_train.sh

fn_cmdline $*

# common parameters
common="${task_mnist} ${loss_classnll}"
outdir="${dir_exp_mnist}/eval_depth"
mkdir -p ${outdir}

# models
conv0="--model forward-network --model-params "
conv1=${conv0}"conv:dims=64,rows=7,cols=7,conn=1,drow=1,dcol=1;act-snorm;"
conv2=${conv1}"conv:dims=64,rows=7,cols=7,conn=8,drow=1,dcol=1;act-snorm;"
conv3=${conv2}"conv:dims=64,rows=5,cols=5,conn=8,drow=1,dcol=1;act-snorm;"
conv4=${conv3}"conv:dims=64,rows=5,cols=5,conn=8,drow=1,dcol=1;act-snorm;"
conv5=${conv4}"conv:dims=64,rows=3,cols=3,conn=8,drow=1,dcol=1;act-snorm;"
conv6=${conv5}"conv:dims=64,rows=3,cols=3,conn=8,drow=1,dcol=1;act-snorm;"

mlp0="--model forward-network --model-params "
mlp1=${mlp0}"affine:dims=128;act-snorm;"
mlp2=${mlp1}"affine:dims=128;act-snorm;"
mlp3=${mlp2}"affine:dims=128;act-snorm;"
mlp4=${mlp3}"affine:dims=128;act-snorm;"
mlp5=${mlp4}"affine:dims=128;act-snorm;"

outlayer="affine:dims=10;act-snorm;"

models=${models}" mlp0 mlp1 mlp2 mlp3 mlp4 mlp5"
models=${models}" conv1 conv2 conv3 conv4 conv5 conv6"

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
        bash $(dirname $0)/plot_models.sh ${outdir}/trial${trial}_all.pdf ${outdir}/trial${trial}_*.state
        bash $(dirname $0)/plot_models.sh ${outdir}/trial${trial}_mlps.pdf ${outdir}/trial${trial}_*mlp*.state
        bash $(dirname $0)/plot_models.sh ${outdir}/trial${trial}_convs.pdf ${outdir}/trial${trial}_*conv*.state
done

fn_sumarize "${outdir}" "${models}" "${trainers}" "${criteria}"
