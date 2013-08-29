task="--task mnist --task-dir /home/cosmin/experiments/databases/mnist/ --loss classnll"
model="--model forward-network --model-params conv:convs=16,crows=8,ccols=8;snorm;max-pool;conv:convs=16,crows=8,ccols=8;snorm;max-pool"

#./build/ncv_trainer ${task} ${model} --trainer batch --trainer-params opt=lbfgs,iter=256,eps=1e-6 --trials 1
./build/ncv_trainer ${task} ${model} --trainer stochastic --trainer-params opt=sgd,epochs=4,eps=1e-6 --trials 1
