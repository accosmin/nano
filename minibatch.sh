params=""
params=${params}" --task mnist --task-dir /home/cosmin/experiments/databases/mnist/"
params=${params}" --loss classnll --model forward-network --model-params conv8x8:convs=16;snorm;conv8x8:convs=16;snorm"
params=${params}" --trainer mini-batch --trainer-params opt=cgd,iter=2,batch=1024,epoch=256 --trials 1"

./build/ncv_trainer ${params}
