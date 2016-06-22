#!/bin/bash

source $(dirname $0)/common_train.sh

# common parameters
common="${task_mnist} ${loss_classnll} --threads ${max_threads}"
outdir="${dir_exp_mnist}/eval_regularizer"
mkdir -p ${outdir}

# models
conv0="--model forward-network --model-params "
conv1=${conv0}"conv:dims=16,rows=9,cols=9,conn=1;act-snorm;"
conv2=${conv1}"conv:dims=32,rows=7,cols=7,conn=2;act-snorm;"
conv3=${conv2}"conv:dims=64,rows=5,cols=5,conn=4;act-snorm;"
conv4=${conv3}"conv:dims=128,rows=3,cols=3,conn=8;act-snorm;"

mlp0="--model forward-network --model-params "
mlp1=${mlp0}"affine:dims=128;act-snorm;"
mlp2=${mlp1}"affine:dims=128;act-snorm;"
mlp3=${mlp2}"affine:dims=128;act-snorm;"
mlp4=${mlp3}"affine:dims=128;act-snorm;"

outlayer="affine:dims=10;"

models=""
models=${models}" mlp0 mlp1 mlp2 mlp3 mlp4"
models=${models}" conv1 conv2 conv3 conv4"

# trainers
epochs=1024
trials=10
fn_make_trainers ${epochs}

trainers="batch_cgd"
#trainers=${trainers}" batch_gd batch_cgd batch_lbfgs"
#trainers=${trainers}" mbatch_gd mbatch_cgd mbatch_lbfgs"
#trainers=${trainers}" stoch_ag stoch_aggr stoch_agfr stoch_adagrad stoch_adadelta stoch_adam stoch_sg stoch_ngd stoch_sgm"

# criteria
criteria="crit_avg crit_l2n crit_var"

# train models
for ((trial=0;trial<${trials};trial++))
do
        for model in ${models}
        do
                for trainer in ${trainers}
                do
                        for criterion in ${criteria}
                        do
                                mfile=${outdir}/trial${trial}_${trainer}_${model}_${criterion}.model
                                sfile=${outdir}/trial${trial}_${trainer}_${model}_${criterion}.state
                                lfile=${outdir}/trial${trial}_${trainer}_${model}_${criterion}.log

                                params="${common} ${!model}${outlayer} ${!trainer} ${!criterion} --model-file ${mfile}"

                                printf "running <%s> ...\n" "${params}"
                                printf "running <%s> ...\n" "${params}" > ${lfile}
                                time ${exe_trainer} ${params} >> ${lfile}
                                printf "\tlog saved to <%s>\n" "${lfile}"
                                printf "\n"
                                printf "\tplotting training evolution ...\n"
                                bash $(dirname $0)/plot_model.sh ${sfile}
                                printf "\n"
                        done
                done
        done
done

exit

# compare
for ((trial=0;trial<${trials};trial++))
for model in ${models}
do
        bash plot_models.sh ${dir_exp_mnist}/${model}.pdf ${dir_exp_mnist}/*_${model}*.state
        bash plot_models.sh ${dir_exp_mnist}/${model}_stoch.pdf ${dir_exp_mnist}/stoch_*_${model}*.state
        bash plot_models.sh ${dir_exp_mnist}/${model}_mbatch.pdf ${dir_exp_mnist}/mbatch_*_${model}*.state
done

# compare models
# bash plot_models.sh ${dir_exp_mnist}/conv_models.pdf ${dir_exp_mnist}/conv_*.state
# bash plot_models.sh ${dir_exp_mnist}/conv_min_models.pdf ${dir_exp_mnist}/conv_min_*.state
# bash plot_models.sh ${dir_exp_mnist}/conv_avg_models.pdf ${dir_exp_mnist}/conv_avg_*.state
