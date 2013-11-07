#!/bin/bash

# paths
dir_results=/home/cosmin/experiments/results
dir_db=/home/cosmin/experiments/databases

trainer=./build/ncv_trainer

common_config="--loss classnll --trainer batch --trainer-params opt=lbfgs,iter=256,eps=1e-6"

# task description = task model model-params trials output
tasks=(
	"mnist forward-network 10 mnist-affine"
	#"mnist forward-network conv:convs=16,crows=8,ccols=8;snorm 10 mnist-hidden1"
	#"mnist forward-network conv:convs=16,crows=8,ccols=8;snorm;conv:convs=16,crows=8,ccols=8;snorm 10 mnist-hidden2"
	
	#"cbcl-faces forward-network 10 cbclfaces-affine"
	#"cbcl-faces forward-network conv:convs=16,crows=8,ccols=8;snorm 10 cbclfaces-hidden1"
	#"cbcl-faces forward-network conv:convs=16,crows=8,ccols=8;snorm;conv:convs=16,crows=8,ccols=8;snorm 10 cbclfaces-hidden2"
	
	#"cifar10 forward-network 10 cifar10-affine"
	#"cifar10 forward-network conv:convs=16,crows=8,ccols=8;snorm 10 cifar10-hidden1"
	#"cifar10 forward-network conv:convs=16,crows=8,ccols=8;snorm;conv:convs=16,crows=8,ccols=8;snorm 10 cifar10-hidden2"
	
	# TODO: STL10
	)

# run tasks
for ((i=0; i<${#tasks[*]}; i++))
do
	task=${tasks[$i]}	
	arr=(${task})

	task_id=${arr[0]}
	model_id=${arr[1]}

	dir_exp=${dir_results}/${task_id}
	mkdir -p ${dir_exp}

	task_dir=${dir_db}/${task_id}

	if [[ ${#arr[*]} -eq 5 ]]
	then 
		model_params=${arr[2]}
		trials=${arr[3]}	
		output=${dir_exp}/${arr[4]}.model
	        log=${dir_exp}/${arr[4]}.log
	else
		model_params=
		trials=${arr[2]}
		output=${dir_exp}/${arr[3]}.model
	        log=${dir_exp}/${arr[3]}.log
	fi	

	if [[ -z ${model_params} ]]
	then
		config="--task ${task_id} --task-dir ${task_dir} --trials ${trials} --model ${model_id} --output ${output} ${common_config}"
	else
		config="--task ${task_id} --task-dir ${task_dir} --trials ${trials} --model ${model_id} --model-params ${model_params} --output ${output} ${common_config}"
	fi
	
	echo "running <${task_id}> ..."
	echo -e "\twith model <${model_id}><${model_params}>"
	echo -e "\twith param <${common_config}>"

	time ${trainer} ${config} > ${log}

	echo -e "\tlog saved to <${log}>"
	echo
done

