common_params="--task mnist --task-dir /home/cosmin/experiments/databases/mnist/ --loss classnll --model convnet --trials 10"

lbfgs_params="--optimizer lbfgs"
sgd_params="--optimizer sgd"
asgd_params="--optimizer asgd"

convlayer_unit_params="8:8:8:unit"
convlayer_tanh_params="8:8:8:tanh"
convlayer_xnorm_params="8:8:8:xnorm"

model_max_layers=1

for (( i=0; i<=${model_max_layers}; i++ ))
do
	convnet_unit_params=network=
	convnet_tanh_params=network=
	convnet_xnorm_params=network=

	for (( j=0; j<${i}; j++ ))
	do
		convnet_unit_params=${convnet_unit_params}${convlayer_unit_params},
		convnet_tanh_params=${convnet_tanh_params}${convlayer_tanh_params},
		convnet_xnorm_params=${convnet_xnorm_params}${convlayer_xnorm_params},
	done

	echo ${convnet_unit_params}	
	time ./build/ncv_trainer ${common_params} ${sgd_params} --model-params ${convnet_unit_params} #> convnet_unit_layers${i}.log

	echo ${convnet_tanh_params}
	time ./build/ncv_trainer ${common_params} ${sgd_params} --model-params ${convnet_tanh_params} > convnet_tanh_layers${i}.log

	echo ${convnet_xnorm_params}
	time ./build/ncv_trainer ${common_params} ${sgd_params} --model-params ${convnet_xnorm_params} > convnet_xnorm_layers${i}.log
done

exit

