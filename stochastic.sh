params=""
params=${params}" --task mnist --task-dir /home/cosmin/experiments/databases/mnist/"
#params=${params}" --loss classnll --model forward-network --model-params conv8x8:convs=16;snorm;conv8x8:convs=8;snorm;conv8x8:convs=8;snorm"
params=${params}" --loss classnll --model forward-network"
params=${params}" --trainer stochastic --trainer-params iters=8096,depth=16 --trials 1"

./build/ncv_trainer ${params}
