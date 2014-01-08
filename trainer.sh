network0=""
network1=${network0}"conv:count=32,rows=8,cols=8;snorm;"
network2=${network1}"conv:count=16,rows=8,cols=8;snorm;"
network3=${network2}"conv:count=8,rows=8,cols=8;snorm;"

model0="--model forward-network"
model1="--model forward-network --model-params ${network1}"
model2="--model forward-network --model-params ${network2}"
model3="--model forward-network --model-params ${network3}"

#trainer="--trainer batch --trainer-params opt=lbfgs,iters=16,eps=1e-6"
trainer="--trainer stochastic --trainer-params opt=asgd,epoch=4"

params=""
params=${params}" --task mnist --task-dir /home/cosmin/experiments/databases/mnist/"
params=${params}" --loss classnll --trials 1 --threads 1"

#valgrind --tool=memcheck --leak-check=yes ./build/ncv_trainer ${params}

echo "training ${model0} ..."
time ./build/ncv_trainer ${params} ${trainer} ${model0} > model0.log

echo "training ${model1} ..."
time ./build/ncv_trainer ${params} ${trainer} ${model1} > model1.log

echo "training ${model2} ..."
time ./build/ncv_trainer ${params} ${trainer} ${model2} > model2.log

echo "training ${model3} ..."
time ./build/ncv_trainer ${params} ${trainer} ${model3} > model3.log

