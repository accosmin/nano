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
task_cbclfaces="--task cbcl-faces --task-dir ${dir_db}/cbcl-faces/"

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

# number of available threads
max_threads=`less /proc/cpuinfo | grep -i processor | wc -l`

# trainers 
stoch_sg="--trainer stochastic --trainer-params opt=sg,epoch=16"
stoch_sga="--trainer stochastic --trainer-params opt=sga,epoch=16"
stoch_sia="--trainer stochastic --trainer-params opt=sia,epoch=16"
stoch_nag="--trainer stochastic --trainer-params opt=nag,epoch=16"

mbatch_lbfgs="--trainer minibatch --trainer-params opt=lbfgs,epoch=1024,batch=1024,ratio=1.1,iters=8,eps=1e-6"
mbatch_cgd="--trainer minibatch --trainer-params opt=cgd,epoch=1024,batch=1024,ratio=1.1,iters=8,eps=1e-6"
mbatch_gd="--trainer minibatch --trainer-params opt=gd,epoch=1024,batch=1024,ratio=1.1,iters=8,eps=1e-6"

batch_lbfgs="--trainer batch --trainer-params opt=lbfgs,iters=1024,eps=1e-6"
batch_cgd="--trainer batch --trainer-params opt=cgd,iters=1024,eps=1e-6"
batch_gd="--trainer batch --trainer-params opt=gd,iters=1024,eps=1e-6"

# criteria
avg_crit="--criterion avg"
l2n_crit="--criterion l2n-reg"
var_crit="--criterion var-reg"

# train a model (results directory, name, parameters)
function fn_train
{
        _mfile=$1/$2.model
        _sfile=$1/$2.state
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
        bash plot_model.sh ${_sfile}
        echo
}  

# train a model (results directory, name, parameters) using callgrind for profiling
function fn_train_callgrind
{
        _args=("$@")
        
        _param="" 
        for ((i=2;i<${#_args[*]};i++))
        do
                _param=${_param}" "${_args[$i]}
        done
        
        echo "running callgrind <${_param}> ..."
        bash ../callgrind.sh ${exe_trainer} ${_param}
        echo
}  
