common_params="--task mnist --task-dir /home/cosmin/experiments/databases/mnist/ --loss classnll --model convnet --trials 1"

lbfgs_params="--optimizer lbfgs"
sgd_params="--optimizer sgd"

convlayer_unit_params="8:8:8:unit"
convlayer_tanh_params="8:8:8:tanh"
convlayer_fun1_params="8:8:8:fun1"
convlayer_fun2_params="8:8:8:fun2"

model_max_layers=2

for (( i=0; i<=${model_max_layers}; i++ ))
do
	convnet_unit_params=network=
	convnet_tanh_params=network=
	convnet_fun1_params=network=
	convnet_fun2_params=network=

	for (( j=0; j<${i}; j++ ))
	do
		convnet_unit_params=${convnet_unit_params}${convlayer_unit_params},
		convnet_tanh_params=${convnet_tanh_params}${convlayer_tanh_params},
		convnet_fun1_params=${convnet_fun1_params}${convlayer_fun1_params},
		convnet_fun2_params=${convnet_fun2_params}${convlayer_fun2_params},
	done

	echo "training <${convnet_unit_params}> ..."
	#time ./build/ncv_trainer ${common_params} ${sgd_params} --model-params ${convnet_unit_params} > convnet_unit_layers${i}_sgd.log
	time ./build/ncv_trainer ${common_params} ${lbfgs_params} --model-params ${convnet_unit_params} #> convnet_unit_layers${i}_lbfgs.log
	exit

	echo "training <${convnet_tanh_params}> ..."
	#time ./build/ncv_trainer ${common_params} ${sgd_params} --model-params ${convnet_tanh_params} > convnet_tanh_layers${i}_sgd.log
	time ./build/ncv_trainer ${common_params} ${lbfgs_params} --model-params ${convnet_tanh_params} > convnet_tanh_layers${i}_lbfgs.log
	
	echo "training <${convnet_fun1_params}> ..."
	#time ./build/ncv_trainer ${common_params} ${sgd_params} --model-params ${convnet_fun1_params} > convnet_fun1_layers${i}_sgd.log
	time ./build/ncv_trainer ${common_params} ${lbfgs_params} --model-params ${convnet_fun1_params} > convnet_fun1_layers${i}_lbfgs.log
	
	echo "training <${convnet_fun2_params}> ..."
	#time ./build/ncv_trainer ${common_params} ${sgd_params} --model-params ${convnet_fun2_params} > convnet_fun2_layers${i}_sgd.log
	time ./build/ncv_trainer ${common_params} ${lbfgs_params} --model-params ${convnet_fun2_params} > convnet_fun2_layers${i}_lbfgs.log
done


