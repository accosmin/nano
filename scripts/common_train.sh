#!/bin/bash

# paths
dir_exp=$HOME/experiments/results
dir_db=$HOME/experiments/databases

exe_trainer=../build-release/apps/ncv_trainer
exe_tester=../build-release/apps/ncv_tester
exe_info=../build-release/apps/ncv_info
exe_max_threads=../build-release/apps/ncv_max_threads

# datasets
dir_db_svhn=${dir_db}/svhn/
dir_db_mnist=${dir_db}/mnist/
dir_db_stl10=${dir_db}/stl10/
dir_db_cifar10=${dir_db}/cifar10/
dir_db_cifar100=${dir_db}/cifar100/
dir_db_norb=${dir_db}/norb/

task_svhn="--task svhn --task-dir ${dir_db_svhn}"
task_mnist="--task mnist --task-dir ${dir_db_mnist}"
task_stl10="--task stl10 --task-dir ${dir_db_stl10}"
task_cifar10="--task cifar10 --task-dir ${dir_db_cifar10}"
task_cifar100="--task cifar100 --task-dir ${dir_db_cifar100}"
task_norb="--task norb --task-dir ${dir_db_norb}"

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
#max_threads=`less /proc/cpuinfo | grep -i processor | wc -l`
max_threads=`${exe_max_threads}`

# trainers 
stoch_ag="--trainer stochastic --trainer-params opt=ag,epoch=32"
stoch_sg="--trainer stochastic --trainer-params opt=sg,epoch=32"
stoch_sga="--trainer stochastic --trainer-params opt=sga,epoch=32"
stoch_sia="--trainer stochastic --trainer-params opt=sia,epoch=32"
stoch_adagrad="--trainer stochastic --trainer-params opt=adagrad,epoch=32"
stoch_adadelta="--trainer stochastic --trainer-params opt=adadelta,epoch=32"

mbatch_lbfgs="--trainer minibatch --trainer-params opt=lbfgs,epoch=32,eps=1e-4"
mbatch_cgd="--trainer minibatch --trainer-params opt=cgd,epoch=32,eps=1e-4"
mbatch_gd="--trainer minibatch --trainer-params opt=gd,epoch=32,eps=1e-4"

batch_lbfgs="--trainer batch --trainer-params opt=lbfgs,iters=1024,eps=1e-4"
batch_cgd="--trainer batch --trainer-params opt=cgd,iters=1024,eps=1e-4"
batch_gd="--trainer batch --trainer-params opt=gd,iters=1024,eps=1e-4"

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
