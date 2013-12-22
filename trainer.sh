network0=""
network1=${network0}"conv8x8:convs=16;snorm;"
network2=${network1}"conv8x8:convs=16;snorm;"
network3=${network2}"conv8x8:convs=16;snorm;"

model="--model forward-network --model-params ${network1}"

#trainer="--trainer batch --trainer-params opt=lbfgs,iters=128,eps=1e-5"
trainer="--trainer stochastic --trainer-params batch=2"

params=""
params=${params}" --task mnist --task-dir /home/cosmin/experiments/databases/mnist/"
params=${params}" --loss logistic --trials 10 --threads 1"
params=${params}" ${trainer} ${model}"

./build/ncv_trainer ${params}
