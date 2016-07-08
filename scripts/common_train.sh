#!/bin/bash

# paths
dir_exp=$HOME/experiments/results
dir_db=$HOME/experiments/databases

exe_trainer=$(dirname $0)/../build-release/apps/train
exe_info=$(dirname $0)/../build-release/apps/info

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

# available trainers (based on the given number of epochs)
function fn_make_trainers
{
        epochs=$1

        stoch_ag="--trainer stochastic --trainer-params opt=ag,epochs=${epochs}"
        stoch_agfr="--trainer stochastic --trainer-params opt=agfr,epochs=${epochs}"
        stoch_aggr="--trainer stochastic --trainer-params opt=aggr,epochs=${epochs}"
        stoch_sg="--trainer stochastic --trainer-params opt=sg,epochs=${epochs}"
        stoch_sgm="--trainer stochastic --trainer-params opt=sgm,epochs=${epochs}"
        stoch_ngd="--trainer stochastic --trainer-params opt=ngd,epochs=${epochs}"
        stoch_adagrad="--trainer stochastic --trainer-params opt=adagrad,epochs=${epochs}"
        stoch_adadelta="--trainer stochastic --trainer-params opt=adadelta,epochs=${epochs}"
        stoch_adam="--trainer stochastic --trainer-params opt=adam,epochs=${epochs}"

        mbatch_lbfgs="--trainer minibatch --trainer-params opt=lbfgs,epochs=${epochs}"
        mbatch_cgd="--trainer minibatch --trainer-params opt=cgd,epochs=${epochs}"
        mbatch_gd="--trainer minibatch --trainer-params opt=gd,epochs=${epochs}"

        batch_lbfgs="--trainer batch --trainer-params opt=lbfgs,epochs=${epochs}"
        batch_cgd="--trainer batch --trainer-params opt=cgd,epochs=${epochs}"
        batch_gd="--trainer batch --trainer-params opt=gd,epochs=${epochs}"
}

# available loss
loss_cauchy="--loss cauchy"
loss_classnll="--loss classnll"
loss_exponential="--loss exponential"
loss_logistic="--loss logistic"
loss_square="--loss square"

# available criteria
crit_avg="--criterion avg"
crit_avg_l2n="--criterion avg-l2n"
crit_avg_var="--criterion avg-var"

crit_max="--criterion max"
crit_max_l2n="--criterion max-l2n"
crit_max_var="--criterion max-var"

