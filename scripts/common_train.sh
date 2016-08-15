#!/bin/bash

# paths
dir_exp=$HOME/experiments/results
dir_db=$HOME/experiments/databases

exe_trainer=$(dirname $0)/../build-release/apps/train
exe_stats=$(dirname $0)/../build-release/apps/stats
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

# default parameters
trials=10
epochs=1000

# usage
function fn_usage
{
        echo "Usage: "
        echo -e "\t--trials     <number of random models to train per configuration>    default=${trials}"
        echo -e "\t--epochs     <number of epochs to train a model>                     default=${epochs}"
        echo
}

# decode command line arguments
function fn_cmdline
{
        while [ "$1" != "" ]
        do
                case $1 in
                        --trials)       shift
                                        trials=$1
                                        ;;
                        --epochs)       shift
                                        epochs=$1
                                        ;;
                        -h | --help)    fn_usage
                                        exit
                                        ;;
                        * )             echo "unrecognized option $1"
                                        echo
                                        fn_usage
                                        exit 1
                esac
                shift
        done
}

# available trainers (based on the given number of epochs)
function fn_make_trainers
{
        local epochs=$1
        local policy=$2

        stoch_ag="--trainer stoch --trainer-params opt=ag,epochs=${epochs},policy=${policy}"
        stoch_agfr="--trainer stoch --trainer-params opt=agfr,epochs=${epochs},policy=${policy}"
        stoch_aggr="--trainer stoch --trainer-params opt=aggr,epochs=${epochs},policy=${policy}"
        stoch_sg="--trainer stoch --trainer-params opt=sg,epochs=${epochs},policy=${policy}"
        stoch_sgm="--trainer stoch --trainer-params opt=sgm,epochs=${epochs},policy=${policy}"
        stoch_ngd="--trainer stoch --trainer-params opt=ngd,epochs=${epochs},policy=${policy}"
        stoch_adagrad="--trainer stoch --trainer-params opt=adagrad,epochs=${epochs},policy=${policy}"
        stoch_adadelta="--trainer stoch --trainer-params opt=adadelta,epochs=${epochs},policy=${policy}"
        stoch_adam="--trainer stoch --trainer-params opt=adam,epochs=${epochs},policy=${policy}"

        batch_lbfgs="--trainer batch --trainer-params opt=lbfgs,epochs=${epochs},policy=${policy}"
        batch_cgd="--trainer batch --trainer-params opt=cgd,epochs=${epochs},policy=${policy}"
        batch_gd="--trainer batch --trainer-params opt=gd,epochs=${epochs},policy=${policy}"
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

