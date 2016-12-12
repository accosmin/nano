#!/bin/bash

source $(dirname $0)/common_train.sh

fn_cmdline $*

# common parameters
common="--task affine --task-params isize=100,osize=10,count=10000,noise=1e-4"

# experimentation directory
outdir="${dir_exp}/affine/eval_trainers"
mkdir -p ${outdir}

# models
mlp0="--model forward-network --model-params "
#mlp1=${mlp0}"affine:dims=100;act-snorm;"
#mlp2=${mlp1}"affine:dims=100;act-snorm;"

outlayer="affine:dims=10;act-snorm;"

mlp0=${mlp0}${outlayer}
#mlp1=${mlp1}${outlayer}
#mlp2=${mlp2}${outlayer}

models=${models}" mlp0"

# losses
losses="loss_square loss_cauchy"

# trainers
fn_make_trainers "stop_early"
trainers=${trainers}" batch_gd batch_cgd batch_lbfgs"
trainers=${trainers}" stoch_sg stoch_sgm stoch_ngd stoch_svrg stoch_asgd"
trainers=${trainers}" stoch_ag stoch_agfr stoch_aggr"
trainers=${trainers}" stoch_adam stoch_adadelta stoch_adagrad"

# criteria
criteria="crit_avg crit_max"

# train models
fn_train "${models}" "${trainers}" "${criteria}" "${losses}"

# compare models
for ((trial=0;trial<${trials};trial++))
do
        for model in ${models}
        do
                for criterion in ${criteria}
                do
                        for loss in ${losses}
                        do
                                bash $(dirname $0)/plot_models.sh \
                                        ${outdir}/trial${trial}_${model}_${criterion}_${loss}_all.pdf \
                                        ${outdir}/trial${trial}_*_${model}_${criterion}_${loss}.state

                                bash $(dirname $0)/plot_models.sh \
                                        ${outdir}/trial${trial}_${model}_${criterion}_${loss}_batch.pdf \
                                        ${outdir}/trial${trial}_batch*_${model}_${criterion}_${loss}.state

                                bash $(dirname $0)/plot_models.sh \
                                        ${outdir}/trial${trial}_${model}_${criterion}_${loss}_stoch.pdf \
                                        ${outdir}/trial${trial}_stoch*_${model}_${criterion}_${loss}.state
                        done
                done
        done
done

fn_sumarize "${outdir}" "${models}" "${trainers}" "${criteria}" "${losses}"
