#!/bin/bash

source $(dirname $0)/common_train.sh

fn_cmdline $*

# common parameters
common="${task_mnist} ${loss_classnll}"
outdir="${dir_exp_mnist}/eval_trainers"
mkdir -p ${outdir}

# models
conv1="--model forward-network --model-params "
conv1=${conv1}"conv:dims=64,rows=7,cols=7,conn=1,drow=2,dcol=2;act-snorm;"
conv1=${conv1}"conv:dims=64,rows=5,cols=5,conn=8,drow=1,dcol=1;act-snorm;"
conv1=${conv1}"conv:dims=64,rows=5,cols=5,conn=8,drow=1,dcol=1;act-snorm;"
conv1=${conv1}"conv:dims=64,rows=3,cols=3,conn=8,drow=1,dcol=1;act-snorm;"

outlayer="affine:dims=10;act-snorm;"

models=${models}" conv1"

# trainers
fn_make_trainers "stop_early"
trainers=${trainers}" batch_gd batch_cgd batch_lbfgs"
trainers=${trainers}" stoch_sg stoch_sgm stoch_ngd"
trainers=${trainers}" stoch_ag stoch_agfr stoch_aggr"
trainers=${trainers}" stoch_adam stoch_adadelta stoch_adagrad"

# criteria
criteria="crit_avg"

# train models
fn_train "${models}" "${trainers}" "${criteria}"

# compare models
for ((trial=0;trial<${trials};trial++))
do
        bash $(dirname $0)/plot_models.sh ${outdir}/trial${trial}_all.pdf ${outdir}/trial${trial}_*.state
        bash $(dirname $0)/plot_models.sh ${outdir}/trial${trial}_batch.pdf ${outdir}/trial${trial}_*batch*.state
        bash $(dirname $0)/plot_models.sh ${outdir}/trial${trial}_stoch.pdf ${outdir}/trial${trial}_*stoch*.state
done

fn_sumarize "${outdir}" "${models}" "${trainers}" "${criteria}"
