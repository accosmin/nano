#!/bin/bash

# paths
dir_exp=/home/cosmin/experiments/results
dir_db=/home/cosmin/experiments/databases

exe_trainer=../build-release/ncv_trainer
exe_tester=../build-release/ncv_tester
exe_info=../build-release/ncv_info

# datasets
task_svhn="--task svhn --task-dir ${dir_db}/svhn/"
task_mnist="--task mnist --task-dir ${dir_db}/mnist/"
task_stl10="--task stl10 --task-dir ${dir_db}/stl10/"
task_cifar10="--task cifar10 --task-dir ${dir_db}/cifar10/"
task_cifar100="--task cifar100 --task-dir ${dir_db}/cifar100/"
task_clbclfaces="--task cbcl-faces --task-dir ${dir_db}/cbcl-faces/"

# results
dir_exp_svhn=${dir_exp}/svhn/
dir_exp_mnist=${dir_exp}/mnist/
dir_exp_stl10=${dir_exp}/stl10/
dir_exp_cifar10=${dir_exp}/cifar10/
dir_exp_cifar100=${dir_exp}/cifar100/
dir_exp_cbclfaces=${dir_exp}/cbcl-faces/

mkdir -p ${dir_exp_svhn}
mkdir -p ${dir_exp_mnist}
mkdir -p ${dir_exp_stl10}
mkdir -p ${dir_exp_cifar10}
mkdir -p ${dir_exp_cifar100}
mkdir -p ${dir_exp_cbclfaces}

#batch="--trainer batch --trainer-params opt=lbfgs,iters=1024,eps=1e-6"
#stochastic="--trainer stochastic --trainer-params opt=sgd,epoch=64"
#minibatch="--trainer minibatch --trainer-params batch=1024,iters=4096,eps=1e-6"

# train a model (model-type, model-parameters, trainer-type, trainer-parameters, configuration-name)
function fn_train
{
        if [[ $# -eq 4 ]]
        then 
                mfile=${dir_exp}/$1-$2-$4.model
                lfile=${dir_exp}/$1-$2-$4.log

                mconfig="--model $1"
                tconfig="--trainer $2 --trainer-params $3"
                pconfig=${param} 
        else
                mfile=${dir_exp}/$1-$3-$5.model
                lfile=${dir_exp}/$1-$3-$5.log

                mconfig="--model $1 --model-params $2"
                tconfig="--trainer $3 --trainer-params $4"
                pconfig=${param}
        fi      
        
        echo "running <${mconfig}> ..."
        echo "running <${tconfig}> ..."
        echo "running <${pconfig}> ..."
        time ${trainer} ${pconfig} ${mconfig} ${tconfig} --output ${mfile} > ${lfile}
        echo -e "\tlog saved to <${lfile}>"
        echo
}  

