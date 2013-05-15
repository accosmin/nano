#!/bin/bash

#######################################################################
# Directories
#######################################################################

dir_exp=/home/cosmin/experiments/results
dir_db=/home/cosmin/experiments/databases

trainer=../build/ncv_trainer

common_config="--loss classnll --iters 8 --eps 1e-5"

tasks="mnist:10 cmu-faces:10 cifar10:10 stl10:1"
models="linear"

for task in ${tasks}
do
	task_name=${task/:*/}
	task_dir=${dir_db}/${task_name}
	trials=${task/*:/}

	dir_results=${dir_exp}/${task_name}
	mkdir -p ${dir_results}
	
	for model in ${models}
	do
		config="--task ${task_name} --task-dir ${task_dir} --trials ${trials} --model ${model} ${common_config}"
		log=${dir_results}/${model}.log

		echo "running <${task_name}> using the model <${model}> ..."
		echo -e "\twith parameters <${config}>"

		time ${trainer} ${config} > ${log}
		
		echo -e "\tlog saved to <${log}>"
		echo
	done
done

