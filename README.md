### Nano

Nano provides numerical optimization and machine learning utilities. For example it can be used to train models such as multi-layer perceptrons (classical neural networks) and convolution networks.


#### Compilation [![build](https://travis-ci.org/accosmin/nano.svg?branch=master)](https://travis-ci.org/accosmin/nano)       [![codecov](https://codecov.io/gh/accosmin/nano/branch/master/graph/badge.svg)](https://codecov.io/gh/accosmin/nano)

Use a C++14 compiler and install Eigen3.3+, LibArchive, Zlib and DevIL. Nano is tested on Linux ([gcc 5+ | clang 3.8+], CMake 3.1+, Ninja or Make) and OSX (XCode 7+, homebrew, CMake 3.1+, Ninja or Make). The code is written to be cross-platform, so it may work (with minor fixes) on other platforms as well (e.g. Windows/MSVC).

The easiest way to compile (and install) is to run:
```
mkdir build-release && cd build-release
cmake -G "Ninja" -DCMAKE_BUILD_TYPE=Release
```
The test programs and utilities will be found in the `build-release` directory.

To build the debugging version with or without address, memory and thread sanitizers (if available) run:
```
mkdir build-debug && cd build-debug
cmake -G "Ninja" -DCMAKE_BUILD_TYPE=Release -DNANO_WITH_[ASAN|MSAN|TSAN]=ON
```

It is recommended to use libc++ with clang by issuing the following command:
```
cmake -G "Ninja" -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER="clang++-4.0" -DNANO_WITH_LIBCPP=ON
```


#### Structure

This library is built around several key concepts mapped to C++ object interfaces. Each object type is registered with an **ID** and thus it can be selected from command line arguments. Also new objects can be easily registered and then they are automatically visible across the library and its associated programs. The list of all supported objects and their parameters is available using:
```
./apps/info --help
```

##### Numerical optimization

The **batch solver** and the **stochastic solver** are gradient-based methods used for minimizing generic multi-dimensional functions. They are suitable for large-scale numerical optimization which are often the product of machine learning problems. Additionally the library provides a large set of unconstrained problems to benchmark the optimization algorithms using for example the following commands:
```
./apps/benchmark_batch --min-dims 10 --max-dims 100 --convex --epsilon 1e-6 --iterations 1000
./apps/benchmark_stoch --min-dims 1 --max-dims 4 --convex --epsilon 1e-4 --epoch-size 100 --epochs 1000
```

The following batch (line-search based) optimization methods are built-in:
```
./apps/info --batch
|---------------|------------------------------------------------|------------------------------------------------------------------|
| batch solvers | description                                    | configuration                                                    |
|---------------|------------------------------------------------|------------------------------------------------------------------|
| cgd           | nonlinear conjugate gradient descent (default) | ls_init=quadratic,ls_strat=interpolation,c1=0.000100,c2=0.100000 |
| cgd-cd        | nonlinear conjugate gradient descent (CD)      | ls_init=quadratic,ls_strat=interpolation,c1=0.000100,c2=0.100000 |
| cgd-dy        | nonlinear conjugate gradient descent (DY)      | ls_init=quadratic,ls_strat=interpolation,c1=0.000100,c2=0.100000 |
| cgd-dycd      | nonlinear conjugate gradient descent (DYCD)    | ls_init=quadratic,ls_strat=interpolation,c1=0.000100,c2=0.100000 |
| cgd-dyhs      | nonlinear conjugate gradient descent (DYHS)    | ls_init=quadratic,ls_strat=interpolation,c1=0.000100,c2=0.100000 |
| cgd-fr        | nonlinear conjugate gradient descent (FR)      | ls_init=quadratic,ls_strat=interpolation,c1=0.000100,c2=0.100000 |
| cgd-hs        | nonlinear conjugate gradient descent (HS)      | ls_init=quadratic,ls_strat=interpolation,c1=0.000100,c2=0.100000 |
| cgd-ls        | nonlinear conjugate gradient descent (LS)      | ls_init=quadratic,ls_strat=interpolation,c1=0.000100,c2=0.100000 |
| cgd-n         | nonlinear conjugate gradient descent (N)       | ls_init=quadratic,ls_strat=interpolation,c1=0.000100,c2=0.100000 |
| cgd-prp       | nonlinear conjugate gradient descent (PRP+)    | ls_init=quadratic,ls_strat=interpolation,c1=0.000100,c2=0.100000 |
| gd            | gradient descent                               | ls_init=quadratic,ls_strat=back-sWolfe,c1=0.000100,c2=0.100000   |
| lbfgs         | limited-memory BFGS                            | ls_init=quadratic,ls_strat=interpolation,c1=0.000100,c2=0.900000 |
|---------------|------------------------------------------------|------------------------------------------------------------------
```

The following stochastic optimization methods are built-in:
```
./apps/info --stoch
|--------------------|--------------------------------------------------------------|---------------|
| stochastic solvers | description                                                  | configuration |
|--------------------|--------------------------------------------------------------|---------------|
| adadelta           | AdaDelta (see citation)                                      |               |
| adagrad            | AdaGrad (see citation)                                       |               |
| adam               | Adam (see citation)                                          |               |
| ag                 | Nesterov's accelerated gradient                              |               |
| agfr               | Nesterov's accelerated gradient with function value restarts |               |
| aggr               | Nesterov's accelerated gradient with gradient restarts       |               |
| asgd               | averaged stochastic gradient (descent)                       |               |
| ngd                | stochastic normalized gradient                               |               |
| rmsprop            | RMSProp (see citation)                                       |               |
| sg                 | stochastic gradient (descent)                                |               |
| sgm                | stochastic gradient (descent) with momentum                  |               |
| svrg               | stochastic variance reduced gradient                         |               |
|--------------------|--------------------------------------------------------------|---------------|
```


##### Machine learning

A **task** describes a classification or regression problem consisting of separate training and test samples (e.g. image patches) with associated target outputs if any. The library has built-in support for various standard benchmark datasets which are loaded directly from the original (compressed) files.
```
./apps/info --task
|---------------|--------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------|
| task          | description                                            | configuration                                                                                                       |
|---------------|--------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------|
| cifar10       | CIFAR-10 (3x32x32 object classification)               | dir=.                                                                                                               |
| cifar100      | CIFAR-100 (3x32x32 object classification)              | dir=.                                                                                                               |
| mnist         | MNIST (1x28x28 digit classification)                   | dir=.                                                                                                               |
| stl10         | STL-10 (3x96x96 semi-supervised object classification) | dir=.                                                                                                               |
| svhn          | SVHN (3x32x32 digit classification in the wild)        | dir=.                                                                                                               |
| synth-charset | synthetic character classification                     | type=digit[lalpha,ualpha,alpha,alphanum],color=rgb[,luma,rgba],irows=32[12,128],icols=32[12,128],count=1000[100,1M] |
|---------------|--------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------|
```

The standard benchmark datasets can be download to $HOME/experiments/databases using:
```
python3 scripts/download_tasks.py
```

The image samples can be saved to disk using for example:
```
./apps/info_task --task mnist --task-params dir=$HOME/experiments/databases/mnist --save-dir ./
```

An **enhancer** is used to access the samples of a task either directly or by performing various transformation like adding random noise or warping the image samples. These transformation are used for improving the robustness of the model and for augmenting the training samples.
```
./apps/info --enhancer
|----------|---------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------|
| enhancer | description                                                               | configuration                                                                                               |
|----------|---------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------|
| default  | use samples as they are                                                   |                                                                                                             |
| noclass  | replace some samples with random samples having no label (classification) | ratio=0.1[0,1],noise=0.1[0,1]                                                                               |
| noise    | add salt&pepper noise to samples                                          | noise=0.1[0,1]                                                                                              |
| warp     | warp image samples (image classification)                                 | type=mixed[translation,rotation,random,mixed],noise=0.1[0,1],sigma=4.0[0,10],alpha=1.0[0,10],beta=1.0[0,10] |
|----------|---------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------|
```

The transformed image samples can be save to disk using for example:
```
./apps/info_enhancer --task mnist --task-params dir=$HOME/experiments/databases/mnist --enhancer warp --save-dir ./
```

A **model** predicts the correct output for a given image patch, either its label (if a classification task) or a score (if a regression task). The feed-forward models can be constructed using a pattern like `[layer_id[:layer_parameters];]+` with the following layers:
```
./apps/info --layer
|-----------|--------------------------------------------------|-------------------------------------------------------------------------------|
| layer     | description                                      | configuration                                                                 |
|-----------|--------------------------------------------------|-------------------------------------------------------------------------------|
| act-sigm  | activation: a(x) = exp(x) / (1 + exp(x))         |                                                                               |
| act-sin   | activation: a(x) = sin(x)                        |                                                                               |
| act-snorm | activation: a(x) = x / sqrt(1 + x^2)             |                                                                               |
| act-splus | activation: a(x) = log(1 + e^x)                  |                                                                               |
| act-tanh  | activation: a(x) = tanh(x)                       |                                                                               |
| act-unit  | activation: a(x) = x                             |                                                                               |
| affine    | transform:  L(x) = A * x + b                     | dims=10[1,4096]                                                               |
| conv      | transform:  L(x) = conv3D(x, kernel) + b         | dims=16[1,256],rows=8[1,32],cols=8[1,32],conn=1[1,16],drow=1[1,8],dcol=1[1,8] |
|-----------|--------------------------------------------------|-------------------------------------------------------------------------------|
```

A **loss** function assigns a scalar score to the prediction of a model `y` by comparing it with the ground truth target `t` (if provided): the lower the score, the better the prediction. The library uses the {-1, +1} codification of class labels.
```
./apps/info --loss
|-------------|----------------------------------------------------------------------------------|---------------|
| loss        | description                                                                      | configuration |
|-------------|----------------------------------------------------------------------------------|---------------|
| cauchy      | multivariate regression:     l(y, t) = log(1 + L2(y, t))                         |               |
| classnll    | single-class classification: l(y, t) = log(y.exp().sum()) + 1/2 * (1 + t).dot(y) |               |
| exponential | multi-class classification:  l(y, t) = exp(-t.dot(y))                            |               |
| logistic    | multi-class classification:  l(y, t) = log(1 + exp(-t.dot(y)))                   |               |
| square      | multivariate regression:     l(y, t) = 1/2 * L2(y, t)                            |               |
|-------------|----------------------------------------------------------------------------------|---------------|
```

A **trainer** optimizes the parameters of a given model to produce the correct outputs for a given task using the cumulated values of a given loss over the training samples as a numerical optimization criteria. All the available trainers tune all their required hyper parameters on a separate validation dataset.
```
./apps/info --trainer
|---------|--------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| trainer | description        | configuration                                                                                                                                                      |
|---------|--------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| batch   | batch trainer      | solver=lbfgs[cgd,cgd-cd,cgd-dy,cgd-dycd,cgd-dyhs,cgd-fr,cgd-hs,cgd-ls,cgd-n,cgd-prp,gd,lbfgs],epochs=1024[4,4096],eps=0.000001,patience=32                         |
| stoch   | stochastic trainer | solver=sg[adadelta,adagrad,adam,ag,agfr,aggr,asgd,ngd,rmsprop,sg,sgm,svrg],epochs=16[1,1024],min_batch=32[32,1024],max_batch=256[32,4096],eps=0.000001,patience=32 |
|---------|--------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------|
```


#### Examples

The library provides various command line programs and utilities. Each program displays its possible arguments with short explanations by running it with `--help`.

Most notably:
* **apps/info** - prints all registered objects with the associated ID and a short description.
* **apps/info_task** - loads a task and prints its detailed description.
* **apps/info_archive** - loads an archive (e.g. .tar, .tgz, .zip) and prints its description (e.g. file names and sizes in bytes).
* **apps/train** - train a model on a given task.
* **apps/evaluate** - test a model on a given task.
* **apps/benchmark_batch** - benchmark all batch optimization methods with varying the line-search parameters on standard test functions.
* **apps/benchmark_stoch** - benchmark all stochastic optimization methods on standard test functions.
* **apps/benchmark_models** - bechmark speed-wise some typical models on a synthetic task.

The `scripts` directory contains examples on how to train various models on different tasks.
