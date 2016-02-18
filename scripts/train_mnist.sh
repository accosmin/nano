#!/bin/bash

source common_train.sh

# common parameters
params=""
params=${params}${task_mnist}
params=${params}" --loss classnll --trials 10 --threads ${max_threads}"

# models
conv="--model forward-network --model-params "
conv=${conv}"conv:dims=16,rows=9,cols=9;act-snorm;pool-max;"
conv=${conv}"conv:dims=32,rows=5,cols=5;act-snorm;pool-max"
conv=${conv}"conv:dims=64,rows=3,cols=3;act-snorm;"

mlp0="--model forward-network --model-params "
mlp1=${mlp0}"affine:dims=128;act-snorm;"
mlp2=${mlp1}"affine:dims=64;act-snorm;"
mlp3=${mlp2}"affine:dims=32;act-snorm;"

outlayer="affine:dims=10;"

# trainers
epochs=1024
fn_trainers ${epochs}

# models
models="mlp0 mlp1 mlp2 mlp3 conv"

# train models
for model in ${models}
do
        trainers="stoch_agfr"
        #trainers=${trainers}" batch_gd batch_cgd batch_lbfgs"
        #trainers=${trainers}" mbatch_gd mbatch_cgd mbatch_lbfgs"
        #trainers=${trainers}" stoch_ag stoch_aggr stoch_agfr stoch_adagrad stoch_adadelta stoch_adam stoch_sg stoch_sgm"

        for trainer in ${trainers}
        do
                fn_train ${dir_exp_mnist} ${trainer}_${model} ${params} ${!trainer} ${avg_crit} ${!model}${outlayer}
                fn_train ${dir_exp_mnist} ${trainer}_${model}_l2n ${params} ${!trainer} ${l2n_crit} ${!model}${outlayer}
                fn_train ${dir_exp_mnist} ${trainer}_${model}_var ${params} ${!trainer} ${var_crit} ${!model}${outlayer}
        done
done

exit

# compare optimizers
for model in ${models}
do
        bash plot_models.sh ${dir_exp_mnist}/${model}.pdf ${dir_exp_mnist}/*_${model}*.state
        bash plot_models.sh ${dir_exp_mnist}/${model}_stoch.pdf ${dir_exp_mnist}/stoch_*_${model}*.state
        bash plot_models.sh ${dir_exp_mnist}/${model}_mbatch.pdf ${dir_exp_mnist}/mbatch_*_${model}*.state
done

# compare models
# bash plot_models.sh ${dir_exp_mnist}/conv_models.pdf ${dir_exp_mnist}/conv_*.state
# bash plot_models.sh ${dir_exp_mnist}/conv_min_models.pdf ${dir_exp_mnist}/conv_min_*.state
# bash plot_models.sh ${dir_exp_mnist}/conv_avg_models.pdf ${dir_exp_mnist}/conv_avg_*.state

