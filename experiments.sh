#!/bin/bash

# paths
dir_results=/home/cosmin/experiments/results
dir_db=/home/cosmin/experiments/databases

exe_trainer=./build/ncv_trainer

common_params="--loss classnll --threads 4"

batch_params="opt=lbfgs,eps=1e-6,iters=1024"
stochastic_params="opt=asgd,epoch=4"

network0=""
network1=${network0}"conv:count=32,rows=8,cols=8;snorm;"
network2=${network1}"conv:count=16,rows=8,cols=8;snorm;"
network3=${network2}"conv:count=8,rows=8,cols=8;snorm;"

# task description = task model [model-params] trainer trainer-params trials output
tasks=(
	#"mnist forward-network ${network0} batch ${batch_params} 10 mnist-network0"	
	#"mnist forward-network ${network1} batch ${batch_params} 10 mnist-network1"
	#"mnist forward-network ${network2} batch ${batch_params} 10 mnist-network2"
	#"mnist forward-network ${network3} batch ${batch_params} 10 mnist-network3"
	
	"mnist forward-network ${network0} stochastic ${stochastic_params} 10 mnist-network0"      
        "mnist forward-network ${network1} stochastic ${stochastic_params} 10 mnist-network1"
        "mnist forward-network ${network2} stochastic ${stochastic_params} 10 mnist-network2"
        "mnist forward-network ${network3} stochastic ${stochastic_params} 10 mnist-network3"
	
	#"cbcl-faces forward-network ${network0} batch ${batch_params} 10 cbclfaces-network0"
	#"cbcl-faces forward-network ${network1} batch ${batch_params} 10 cbclfaces-network1"
	#"cbcl-faces forward-network ${network2} batch ${batch_params} 10 cbclfaces-network2"
	
	#"cifar10 forward-network ${network0} batch ${batch_params} 10 cifar10-network0"
	#"cifar10 forward-network ${network1} batch ${batch_params} 10 cifar10-network1"
	#"cifar10 forward-network ${network2} batch ${batch_params} 10 cifar10-network2"
	#"cifar10 forward-network ${network3} batch ${batch_params} 10 cifar10-network3"
	
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

	if [[ ${#arr[*]} -eq 7 ]]
	then 
		model_params=${arr[2]}
		trainer_id=${arr[3]}
		trainer_params=${arr[4]}
		trials=${arr[5]}	
		output=${dir_exp}/${arr[6]}.model
	        log=${dir_exp}/${arr[6]}.log
	else
		model_params=
		trainer_id=${arr[2]}
                trainer_params=${arr[3]}
                trials=${arr[4]}        
                output=${dir_exp}/${arr[5]}.model
                log=${dir_exp}/${arr[5]}.log
	fi	

	config=""
	config=${config}" --task ${task_id} --task-dir ${task_dir}"
	config=${config}" --trials ${trials} --output ${output} ${common_params}"                
	if [[ -z ${model_params} ]]
	then
		config=${config}" --model ${model_id}"
	else
                config=${config}" --model ${model_id} --model-params ${model_params}"
	fi
	config=${config}" --trainer ${trainer_id} --trainer-params ${trainer_params}"
	
	echo "running <${task_id}> ..."
	echo -e "\twith model   <${model_id}><${model_params}>"
        echo -e "\twith trainer <${trainer_id}><${trainer_params}>"
	echo -e "\twith param   <${common_params}>"

	time ${exe_trainer} ${config} > ${log}

	echo -e "\tlog saved to <${log}>"
	echo
done

