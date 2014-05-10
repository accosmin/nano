#!/bin/bash

source common.sh

# common parameters
params=""
params=${params}"--loss classdif --trials 10 --threads 1"

#batch="--trainer batch --trainer-params opt=lbfgs,iters=1024,eps=1e-6"
#stochastic="--trainer stochastic --trainer-params opt=sgd,epoch=64"

# trainers (minibatch configurations to evaluate)
minibatch_none="--trainer minibatch --trainer-params batch=1024,iters=256,eps=1e-6,reg=none"
minibatch_l2nm="--trainer minibatch --trainer-params batch=1024,iters=256,eps=1e-6,reg=l2"
minibatch_vari="--trainer minibatch --trainer-params batch=1024,iters=256,eps=1e-6,reg=var"

batch_none="--trainer batch --trainer-params opt=lbfgs,iters=32,eps=1e-6,reg=none"
batch_l2nm="--trainer batch --trainer-params opt=lbfgs,iters=32,eps=1e-6,reg=l2"
batch_vari="--trainer batch --trainer-params opt=lbfgs,iters=32,eps=1e-6,reg=var"

trainers=("minibatch_none#${minibatch_none}"
        "minibatch_l2nm#${minibatch_l2nm}"
        "minibatch_vari#${minibatch_vari}")
        
#       "batch_none#${batch_none}"
#       "batch_l2nm#${batch_l2nm}"
#       "batch_vari#${batch_vari}")
 
# models
conv0="--model forward-network --model-params "
conv1=${conv0}"conv:dims=32,rows=7,cols=7;snorm;pool-abs;"
conv2=${conv1}"conv:dims=32,rows=4,cols=4;snorm;pool-abs;"
conv3=${conv2}"conv:dims=32,rows=4,cols=4;snorm;"

mlp0="--model forward-network --model-params "
mlp1=${mlp0}"linear:dims=100;snorm;"
mlp2=${mlp1}"linear:dims=100;snorm;"
mlp3=${mlp2}"linear:dims=100;snorm;"
mlp4=${mlp3}"linear:dims=100;snorm;"

outlayer=";linear:dims=10;norm-max:type=global;"

models=("mlp0#${mlp0}${outlayer}"
	"mlp1#${mlp1}${outlayer}"
	"mlp2#${mlp2}${outlayer}"
	"mlp3#${mlp3}${outlayer}"
	"mlp4#${mlp4}${outlayer}")
#	"conv1#${conv1};${outlayer}"
#	"conv2#${conv2};${outlayer}"
#	"conv3#${conv3};${outlayer}")

# train models
for ((i=0;i<${#trainers[*]};i++))
{
        trainer=${trainers[$i]}
        tname=${trainer//#*/}
        tparams=${trainer/*#/}
        
        for ((j=0;j<${#models[*]};j++))
        {
                model=${models[$j]}
                mname=${model//#*/}
                mparams=${model/*#/}
                
                mfile=${dir_exp_mnist}/test-${tname}-${mname}.model
                lfile=${dir_exp_mnist}/test-${tname}-${mname}.log
        
                echo "using trainer <${tparams}> ..."
                echo "using model <${mparams}> ..."
                echo "using task <${task_mnist}> ..."
                echo "using parameters <${params}> ..." 
                echo "saving model <${mfile}> ..."
                echo "saving log <${lfile}> ..."
                time ${exe_trainer} ${tparams} ${mparams} ${params} ${task_mnist} --output ${mfile} > ${lfile}
                echo -e "\tlog saved to <${lfile}>"
                echo
        }        
}
