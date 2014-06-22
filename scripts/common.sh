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

# number of available threads
max_threads=`less /proc/cpuinfo | grep -i processor | wc -l`

# train a model (results directory, name, parameters)
function fn_train
{
        _mfile=$1/$2.model
        _lfile=$1/$2.log
        _args=("$@")
        
        _param="" 
        for ((i=2;i<${#_args[*]};i++))
        do
                _param=${_param}" "${_args[$i]}
        done
        
        echo "running <${_param}> ..."
        echo "running <${_param}> ..." > ${_lfile}        
        time ${exe_trainer} ${_param} --output ${_mfile} >> ${_lfile}     
        echo -e "\tlog saved to <${_lfile}>"
        echo
        echo -e "\tplotting optimization states ..."
        for _sfile in $1/$2_*.state
        do
                bash plot_trainlog.sh ${_sfile}
        done
        echo
}  

# train a model (results directory, name, parameters) using callgrind for profiling
function fn_train_callgrind
{
        _mfile=$1/$2.model
        _lfile=$1/$2.log
        _args=("$@")
        
        _param="" 
        for ((i=2;i<${#_args[*]};i++))
        do
                _param=${_param}" "${_args[$i]}
        done
        
        echo "running <${_param}> ..."
        echo "running <${_param}> ..." > ${_lfile}        
        bash ../callgrind.sh ${exe_trainer} ${_param} --output ${_mfile} >> ${_lfile}    
        echo -e "\tlog saved to <${_lfile}>"
        echo
}  
