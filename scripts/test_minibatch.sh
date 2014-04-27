#!/bin/bash

source common.sh

# common parameters
params=""
params=${params}"--loss classnll --trials 10 --threads 2"

#batch="--trainer batch --trainer-params opt=lbfgs,iters=1024,eps=1e-6"
#stochastic="--trainer stochastic --trainer-params opt=sgd,epoch=64"

# trainers (minibatch configurations to evaluate)
minibatch128="--trainer minibatch --trainer-params batch=128,iters=128,eps=1e-6"
minibatch256="--trainer minibatch --trainer-params batch=256,iters=128,eps=1e-6"
minibatch512="--trainer minibatch --trainer-params batch=512,iters=128,eps=1e-6"
minibatch1024="--trainer minibatch --trainer-params batch=1024,iters=128,eps=1e-6"
minibatch2048="--trainer minibatch --trainer-params batch=2048,iters=128,eps=1e-6"

trainers=("minibatch128#${minibatch128}"
        "minibatch256#${minibatch256}"
        "minibatch512#${minibatch512}"
        "minibatch1024#${minibatch1024}"
        "minibatch2048#${minibatch2048}")
 
# models
conv0="--model forward-network --model-params "
conv1=${conv0}"conv:dims=32,rows=7,cols=7;snorm;smax-abs-pool;"
conv2=${conv1}"conv:dims=32,rows=4,cols=4;snorm;smax-abs-pool;"
conv3=${conv2}"conv:dims=32,rows=4,cols=4;snorm;"

mlp0="--model forward-network --model-params "
mlp1=${mlp0}"linear:dims=100;snorm;"
mlp2=${mlp1}"linear:dims=100;snorm;"
mlp3=${mlp2}"linear:dims=100;snorm;"
mlp4=${mlp3}"linear:dims=100;snorm;"

models=("conv1#${conv1}"
      "conv2#${conv2}"
      "conv3#${conv3}"
      "mlp0#${mlp0}"
      "mlp1#${mlp1}"
      "mlp2#${mlp2}"
      "mlp3#${mlp3}"
      "mlp4#${mlp4}")

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
