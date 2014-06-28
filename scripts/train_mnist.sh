#!/bin/bash

source common.sh

# common parameters
params=""
params=${params}${task_mnist}
params=${params}" --loss class-ratio --trials 1 --threads 8"

# trainers 
stoch_sg="--trainer stochastic --trainer-params opt=sg,epoch=8"
stoch_sga="--trainer stochastic --trainer-params opt=sga,epoch=256"
stoch_sia="--trainer stochastic --trainer-params opt=sia,epoch=256"

mbatch_lbfgs="--trainer minibatch --trainer-params opt=lbfgs,epoch=2048,batch=1024,iters=16,eps=1e-6"
mbatch_cgd="--trainer minibatch --trainer-params opt=cgd,epoch=2048,batch=1024,iters=16,eps=1e-6"
mbatch_gd="--trainer minibatch --trainer-params opt=gd,epoch=16,batch=64,iters=4,eps=1e-6"

batch_lbfgs="--trainer batch --trainer-params opt=lbfgs,iters=2048,eps=1e-6"
batch_cgd="--trainer batch --trainer-params opt=cgd,iters=2048,eps=1e-6"
batch_gd="--trainer batch --trainer-params opt=gd,iters=2048,eps=1e-6"

# models
conv1_max="--model forward-network --model-params "
conv1_max=${conv1_max}"conv:dims=16,rows=7,cols=7;act-snorm;pool-max;"
conv1_max=${conv1_max}"conv:dims=16,rows=4,cols=4;act-snorm;pool-max;"
conv1_max=${conv1_max}"conv:dims=16,rows=4,cols=4;act-snorm;"

conv1_min=${conv1_max//pool-max/pool-min}
conv1_avg=${conv1_max//pool-max/pool-avg}
        
conv2_max="--model forward-network --model-params "
conv2_max=${conv2_max}"conv:dims=16,rows=7,cols=7;act-snorm;pool-max;"
conv2_max=${conv2_max}"conv:dims=16,rows=6,cols=6;act-snorm;pool-max;"
conv2_max=${conv2_max}"conv:dims=16,rows=3,cols=3;act-snorm;"

conv2_min=${conv2_max//pool-max/pool-min}
conv2_avg=${conv2_max//pool-max/pool-avg}
       
conv3_max="--model forward-network --model-params "
conv3_max=${conv3_max}"conv:dims=16,rows=5,cols=5;act-snorm;pool-max;"
conv3_max=${conv3_max}"conv:dims=16,rows=5,cols=5;act-snorm;pool-max;"
conv3_max=${conv3_max}"conv:dims=16,rows=4,cols=4;act-snorm;"

conv3_min=${conv3_max//pool-max/pool-min}
conv3_avg=${conv3_max//pool-max/pool-avg}

mlp0="--model forward-network --model-params "
mlp1=${mlp0}"linear:dims=64;act-snorm;"
mlp2=${mlp1}"linear:dims=64;act-snorm;"
mlp3=${mlp2}"linear:dims=64;act-snorm;"
mlp4=${mlp3}"linear:dims=64;act-snorm;"
mlp5=${mlp4}"linear:dims=64;act-snorm;"
mlp6=${mlp5}"linear:dims=64;act-snorm;"

outlayer=";linear:dims=10;softmax:type=global;"

# train models
for trainer in `echo "mbatch_gd"` #"stoch_sg stoch_sga stoch_sia mbatch_gd mbatch_cgd mbatch_lbfgs batch_gd batch_cgd batch_lbfgs"`
do
        #fn_train ${dir_exp_mnist} mlp0_${trainer} ${params} ${!trainer} ${mlp0}${outlayer}
        #fn_train ${dir_exp_mnist} mlp1_${trainer} ${params} ${!trainer} ${mlp1}${outlayer}
        #fn_train ${dir_exp_mnist} mlp2_${trainer} ${params} ${!trainer} ${mlp2}${outlayer}
        #fn_train ${dir_exp_mnist} mlp3_${trainer} ${params} ${!trainer} ${mlp3}${outlayer}
        
        fn_train ${dir_exp_mnist} conv1_max_${trainer} ${params} ${!trainer} ${conv1_max}${outlayer}
        fn_train ${dir_exp_mnist} conv1_min_${trainer} ${params} ${!trainer} ${conv1_min}${outlayer}
        fn_train ${dir_exp_mnist} conv1_avg_${trainer} ${params} ${!trainer} ${conv1_avg}${outlayer}

#         fn_train ${dir_exp_mnist} conv2_max_${trainer} ${params} ${!trainer} ${conv2_max}${outlayer}
#         fn_train ${dir_exp_mnist} conv2_min_${trainer} ${params} ${!trainer} ${conv2_min}${outlayer}
#         fn_train ${dir_exp_mnist} conv2_avg_${trainer} ${params} ${!trainer} ${conv2_avg}${outlayer}
#         
#         fn_train ${dir_exp_mnist} conv3_max_${trainer} ${params} ${!trainer} ${conv3_max}${outlayer}
#         fn_train ${dir_exp_mnist} conv3_min_${trainer} ${params} ${!trainer} ${conv3_min}${outlayer}
#         fn_train ${dir_exp_mnist} conv3_avg_${trainer} ${params} ${!trainer} ${conv3_avg}${outlayer}
done

# compare models
bash plot_models.sh ${dir_exp_mnist}/models.pdf ${dir_exp_mnist}/*.state