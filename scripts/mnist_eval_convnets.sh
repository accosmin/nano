#!/bin/bash

source $(dirname $0)/common_train.sh

fn_cmdline $*

# common parameters
common="${task_mnist} ${loss_classnll}"
outdir="${dir_exp_mnist}/eval_convnets"
mkdir -p ${outdir}

# models
# * vary input-output connectivity {4, 8}
# * vary stride {1, 2}
convc8d1="--model forward-network --model-params "
convc8d1=${convc8d1}"conv:dims=64,rows=7,cols=7,conn=1,drow=1,dcol=1;act-snorm;"
convc8d1=${convc8d1}"conv:dims=64,rows=7,cols=7,conn=8,drow=1,dcol=1;act-snorm;"
convc8d1=${convc8d1}"conv:dims=64,rows=5,cols=5,conn=8,drow=1,dcol=1;act-snorm;"
convc8d1=${convc8d1}"conv:dims=64,rows=5,cols=5,conn=8,drow=1,dcol=1;act-snorm;"
convc8d1=${convc8d1}"conv:dims=64,rows=5,cols=5,conn=8,drow=1,dcol=1;act-snorm;"
convc8d1=${convc8d1}"conv:dims=64,rows=3,cols=3,conn=8,drow=1,dcol=1;act-snorm;"

convc4d1="--model forward-network --model-params "
convc4d1=${convc4d1}"conv:dims=64,rows=7,cols=7,conn=1,drow=1,dcol=1;act-snorm;"
convc4d1=${convc4d1}"conv:dims=64,rows=7,cols=7,conn=4,drow=1,dcol=1;act-snorm;"
convc4d1=${convc4d1}"conv:dims=64,rows=5,cols=5,conn=4,drow=1,dcol=1;act-snorm;"
convc4d1=${convc4d1}"conv:dims=64,rows=5,cols=5,conn=4,drow=1,dcol=1;act-snorm;"
convc4d1=${convc4d1}"conv:dims=64,rows=5,cols=5,conn=4,drow=1,dcol=1;act-snorm;"
convc4d1=${convc4d1}"conv:dims=64,rows=3,cols=3,conn=4,drow=1,dcol=1;act-snorm;"

convc8d2="--model forward-network --model-params "
convc8d2=${convc8d2}"conv:dims=64,rows=7,cols=7,conn=1,drow=2,dcol=2;act-snorm;"
convc8d2=${convc8d2}"conv:dims=64,rows=5,cols=5,conn=8,drow=1,dcol=1;act-snorm;"
convc8d2=${convc8d2}"conv:dims=64,rows=3,cols=3,conn=8,drow=1,dcol=1;act-snorm;"

convc4d2="--model forward-network --model-params "
convc4d2=${convc4d2}"conv:dims=64,rows=7,cols=7,conn=1,drow=2,dcol=2;act-snorm;"
convc4d2=${convc4d2}"conv:dims=64,rows=5,cols=5,conn=4,drow=1,dcol=1;act-snorm;"
convc4d2=${convc4d2}"conv:dims=64,rows=3,cols=3,conn=4,drow=1,dcol=1;act-snorm;"

outlayer="affine:dims=10;act-snorm;"

models=${models}" convc8d1 convc8d2 convc4d1 convc4d2"

# trainers
fn_make_trainers "stop_early"
trainers="stoch_svrg"

# criteria
criteria="crit_avg"

# train models
fn_train "${models}" "${trainers}" "${criteria}"

# compare models
for ((trial=0;trial<${trials};trial++))
do
        bash $(dirname $0)/plot_models.sh ${outdir}/trial${trial}_all.pdf ${outdir}/trial${trial}_*.state
        bash $(dirname $0)/plot_models.sh ${outdir}/trial${trial}_convs_c8.pdf ${outdir}/trial${trial}_*convc8*.state
        bash $(dirname $0)/plot_models.sh ${outdir}/trial${trial}_convs_c4.pdf ${outdir}/trial${trial}_*convc4*.state
done

fn_sumarize "${outdir}" "${models}" "${trainers}" "${criteria}"
