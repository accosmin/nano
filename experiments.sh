#!/bin/bash

echo "This script is not updated with the new interface/models!"
exit

# paths
dir_exp=/home/cosmin/experiments/results
dir_db=/home/cosmin/experiments/databases

trainer=../build/ncv_trainer

common_config="--loss classnll"

# task description = task_id:model_id:model_parameters:number_of_trials
tasks=(
	"mnist:affine:proc=luma:10"
	"cmu-faces:affine:proc=luma:10"
	"cifar10:affine:proc=rgba:10"
	"stl10:affine:proc=rgba:1"
	)

# run tasks
for ((i=0; i<${#tasks[*]}; i++))
do
	task=${tasks[$i]}
	task_spaces=${task//:/ }

	arr=(${task//:/ })
	task_id=${arr[0]}
	model_id=${arr[1]}
	model_params=${arr[2]}
	trials=${arr[3]}

	dir_results=${dir_exp}/${task_name}
	mkdir -p ${dir_results}

	task_dir=${dir_db}/${task_id}

	config="--task ${task_id} --task-dir ${task_dir} --trials ${trials} --model ${model_id} --model-params ${model_params} ${common_config}"
	log=${dir_results}/${model_id}.log

	echo "running <${task_id}> using the model <${model_id}> ..."
	echo -e "\twith parameters <${config}>"

	#time ${trainer} ${config} > ${log}

	echo -e "\tlog saved to <${log}>"
	echo
done

