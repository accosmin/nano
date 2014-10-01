#!/bin/bash

source common.sh

# common parameters
params=""
params=${params}${task_mnist}
params=${params}" --loss classnll --trials 10 --threads 8"

# trainers 
stoch_sg="--trainer stochastic --trainer-params opt=sg,epoch=16"
stoch_sga="--trainer stochastic --trainer-params opt=sga,epoch=16"
stoch_sia="--trainer stochastic --trainer-params opt=sia,epoch=16"

mbatch_lbfgs="--trainer minibatch --trainer-params opt=lbfgs,epoch=256,batch=1024,iters=8,eps=1e-6"
mbatch_cgd="--trainer minibatch --trainer-params opt=cgd,epoch=2048,batch=1024,iters=16,eps=1e-6"
mbatch_gd="--trainer minibatch --trainer-params opt=gd,epoch=2048,batch=64,iters=4,eps=1e-6"

batch_lbfgs="--trainer batch --trainer-params opt=lbfgs,iters=2048,eps=1e-6"
batch_cgd="--trainer batch --trainer-params opt=cgd,iters=2048,eps=1e-6"
batch_gd="--trainer batch --trainer-params opt=gd,iters=2048,eps=1e-6"

# criteria
avg_crit="--criterion avg"
l2n_crit="--criterion l2n-reg"
var_crit="--criterion var-reg"

# models
conv_max="--model forward-network --model-params "
conv_max=${conv_max}"conv:dims=16,rows=6,cols=6;act-snorm;pool-max;"
conv_max=${conv_max}"conv:dims=32,rows=5,cols=5;act-snorm;pool-max;"
conv_max=${conv_max}"conv:dims=64,rows=4,cols=4;act-snorm;"

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
for model in `echo "conv_max conv_avg conv_min"`
do
        for trainer in `echo "stoch_sg stoch_sga stoch_sia"` # mbatch_gd mbatch_cgd mbatch_lbfgs batch_gd batch_cgd batch_lbfgs"`
        do
                fn_train ${dir_exp_mnist} ${model} ${params} ${!trainer} ${avg_crit} ${!model}${outlayer}
                fn_train ${dir_exp_mnist} ${model}_l2n ${params} ${!trainer} ${l2n_crit} ${!model}${outlayer}
                fn_train ${dir_exp_mnist} ${model}_var ${params} ${!trainer} ${var_crit} ${!model}${outlayer}
        done
done

exit

# compare models
bash plot_models.sh ${dir_exp_mnist}/models.pdf ${dir_exp_mnist}/*.state

bash plot_models.sh ${dir_exp_mnist}/conv_max_models.pdf ${dir_exp_mnist}/conv_max_*.state
bash plot_models.sh ${dir_exp_mnist}/conv_min_models.pdf ${dir_exp_mnist}/conv_min_*.state
bash plot_models.sh ${dir_exp_mnist}/conv_avg_models.pdf ${dir_exp_mnist}/conv_avg_*.state

