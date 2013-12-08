model="--model forward-network"
model="--model forward-network --model-params conv8x8:convs=16;snorm"
#model="--model forward-network --model-params conv8x8:convs=16;snorm;conv8x8:convs=8;snorm"
#model="--model forward-network --model-params conv8x8:convs=16;snorm;conv8x8:convs=8;snorm;conv8x8:convs=8;snorm"

#trainer="--trainer batch --trainer-params opt=lbfgs,iters=128,eps=1e-5"
#trainer="--trainer minibatch --trainer-params opt=cgd,iters=2,batch=1024,epoch=256"
trainer="--trainer stochastic --trainer-params batch=8"

params=""
params=${params}" --task mnist --task-dir /home/cosmin/experiments/databases/mnist/"
params=${params}" --loss classnll --trials 1 --threads 1"
params=${params}" ${trainer} ${model}"

./build/ncv_trainer ${params}
