#!/bin/bash

# paths
dir_exp=$HOME/experiments/results
dir_db=$HOME/experiments/databases

exe_trainer=$(dirname $0)/../build-release/apps/train
exe_tester=$(dirname $0)/../build-release/apps/evaluate
exe_info=$(dirname $0)/../build-release/apps/info
exe_max_threads=$(dirname $0)/../build-release/apps/max_threads

# available datasets
dir_db_svhn=${dir_db}/svhn/
dir_db_mnist=${dir_db}/mnist/
dir_db_stl10=${dir_db}/stl10/
dir_db_cifar10=${dir_db}/cifar10/
dir_db_cifar100=${dir_db}/cifar100/

task_svhn="--task svhn --task-params dir=${dir_db_svhn}"
task_mnist="--task mnist --task-params dir=${dir_db_mnist}"
task_stl10="--task stl10 --task-params dir=${dir_db_stl10}"
task_cifar10="--task cifar10 --task-params dir=${dir_db_cifar10}"
task_cifar100="--task cifar100 --task-params dir=${dir_db_cifar100}"

# results
dir_exp_svhn=${dir_exp}/svhn/
dir_exp_mnist=${dir_exp}/mnist/
dir_exp_stl10=${dir_exp}/stl10/
dir_exp_cifar10=${dir_exp}/cifar10/
dir_exp_cifar100=${dir_exp}/cifar100/

mkdir -p ${dir_exp_svhn}
mkdir -p ${dir_exp_mnist}
mkdir -p ${dir_exp_stl10}
mkdir -p ${dir_exp_cifar10}
mkdir -p ${dir_exp_cifar100}

# number of available threads
#max_threads=$(less /proc/cpuinfo | grep -i processor | wc -l)
max_threads=$(${exe_max_threads})

# available trainers (based on the given number of epochs)
function fn_make_trainers
{
        epochs=$1

        stoch_ag="--trainer stochastic --trainer-params opt=ag,epoch=${epochs}"
        stoch_agfr="--trainer stochastic --trainer-params opt=agfr,epoch=${epochs}"
        stoch_aggr="--trainer stochastic --trainer-params opt=aggr,epoch=${epochs}"
        stoch_sg="--trainer stochastic --trainer-params opt=sg,epoch=${epochs}"
        stoch_sgm="--trainer stochastic --trainer-params opt=sgm,epoch=${epochs}"
        stoch_adagrad="--trainer stochastic --trainer-params opt=adagrad,epoch=${epochs}"
        stoch_adadelta="--trainer stochastic --trainer-params opt=adadelta,epoch=${epochs}"
        stoch_adam="--trainer stochastic --trainer-params opt=adam,epoch=${epochs}"

        mbatch_lbfgs="--trainer minibatch --trainer-params opt=lbfgs,epoch=${epochs},eps=1e-4"
        mbatch_cgd="--trainer minibatch --trainer-params opt=cgd,epoch=${epochs},eps=1e-4"
        mbatch_gd="--trainer minibatch --trainer-params opt=gd,epoch=${epochs},eps=1e-4"

        batch_lbfgs="--trainer batch --trainer-params opt=lbfgs,iters=${epochs},eps=1e-4"
        batch_cgd="--trainer batch --trainer-params opt=cgd,iters=${epochs},eps=1e-4"
        batch_gd="--trainer batch --trainer-params opt=gd,iters=${epochs},eps=1e-4"
}

# available loss
loss_cauchy="--loss cauchy"
loss_classnll="--loss classnll"
loss_exponential="--loss exponential"
loss_logistic="--loss logistic"
loss_square="--loss square"

# available criteria
crit_avg="--criterion avg"
crit_l2n="--criterion l2n-reg"
crit_var="--criterion var-reg"

