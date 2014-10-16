#!/bin/bash

source common.sh

# common parameters
params=""
params=${params}${task_mnist}
params=${params}" --loss classnll --trials 10 --threads ${max_threads}"

# models
conv_max="--model forward-network --model-params "
conv_max=${conv_max}"conv:dims=16,rows=6,cols=6,type=full;act-snorm;pool-max;"
conv_max=${conv_max}"conv:dims=32,rows=5,cols=5,type=full;act-snorm;pool-max;"
conv_max=${conv_max}"conv:dims=64,rows=4,cols=4,type=full;act-snorm;"

conv_min=${conv_max//pool-max/pool-min}
conv_avg=${conv_max//pool-max/pool-avg}

rconv_max=${conv_max//type=full/type=rand}
rconv_min=${conv_min//type=full/type=rand}
rconv_avg=${conv_avg//type=full/type=rand}

mconv_max=${conv_max//type=full/type=mask}
mconv_min=${conv_min//type=full/type=mask}
mconv_avg=${conv_avg//type=full/type=mask}
        
mlp0="--model forward-network --model-params "
mlp1=${mlp0}"linear:dims=128;act-snorm;"
mlp2=${mlp1}"linear:dims=64;act-snorm;"
mlp3=${mlp2}"linear:dims=32;act-snorm;"

outlayer="linear:dims=10;"

# train models
for model in `echo "conv_max conv_avg conv_min rconv_max rconv_avg rconv_min mconv_max mconv_avg mconv_min mlp0 mlp1 mlp2 mlp3"`
do
        for trainer in `echo "mbatch_lbfgs"`
	do
	        fn_train ${dir_exp_mnist} ${trainer}_${model} ${params} ${!trainer} ${avg_crit} ${!model}${outlayer}
	done

        #for trainer in `echo "mbatch_gd mbatch_cgd mbatch_lbfgs"` # batch_gd batch_cgd batch_lbfgs"`
        #do
        #        fn_train ${dir_exp_mnist} ${trainer}_${model} ${params} ${!trainer} ${avg_crit} ${!model}${outlayer}
        #        fn_train ${dir_exp_mnist} ${trainer}_${model}_l2n ${params} ${!trainer} ${l2n_crit} ${!model}${outlayer}
        #        fn_train ${dir_exp_mnist} ${trainer}_${model}_var ${params} ${!trainer} ${var_crit} ${!model}${outlayer}
        #done
        #for trainer in `echo "stoch_sg stoch_sga stoch_sia"`
        #do
        #        fn_train ${dir_exp_mnist} ${trainer}_${model} ${params} ${!trainer} ${avg_crit} ${!model}${outlayer}
        #        fn_train ${dir_exp_mnist} ${trainer}_${model}_l2n ${params} ${!trainer} ${l2n_crit} ${!model}${outlayer}
        #done
done

exit

# compare models
bash plot_models.sh ${dir_exp_mnist}/models.pdf ${dir_exp_mnist}/*.state

bash plot_models.sh ${dir_exp_mnist}/conv_max_models.pdf ${dir_exp_mnist}/conv_max_*.state
bash plot_models.sh ${dir_exp_mnist}/conv_min_models.pdf ${dir_exp_mnist}/conv_min_*.state
bash plot_models.sh ${dir_exp_mnist}/conv_avg_models.pdf ${dir_exp_mnist}/conv_avg_*.state

