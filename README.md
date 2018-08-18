### Nano

Nano provides numerical optimization and machine learning utilities. For example it can be used to train models such as multi-layer perceptrons (classical neural networks) and convolution networks.


#### Compilation [![build](https://travis-ci.org/accosmin/nano.svg?branch=master)](https://travis-ci.org/accosmin/nano) [![codecov](https://codecov.io/gh/accosmin/nano/branch/master/graph/badge.svg)](https://codecov.io/gh/accosmin/nano)

Use a C++14 compiler and install LibArchive, Zlib and DevIL. Nano is tested on Linux ([gcc 5+ | clang 3.8+], CMake 3.1+, Ninja or Make) and OSX (XCode 7+, homebrew, CMake 3.1+, Ninja or Make). The code is written to be cross-platform, so it may work (with minor fixes) on other platforms as well (e.g. Windows/MSVC).

The easiest way to compile (and install) is to run:
```
mkdir build-release && cd build-release
cmake -G "Ninja" -DCMAKE_BUILD_TYPE=Release -DNANO_WITH_BENCH=ON
```
The test programs and utilities will be found in the `build-release` directory.

To build the debugging version with or without address, memory and thread sanitizers (if available) run:
```
mkdir build-debug && cd build-debug
cmake -G "Ninja" -DCMAKE_BUILD_TYPE=Debug -DNANO_WITH_[ASAN|LSAN|USAN|MSAN|TSAN]=ON
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

The **solver** is a gradient-based method for minimizing generic multi-dimensional functions. They are suitable for large-scale numerical optimization which are often the product of machine learning problems. Additionally the library provides a large set of unconstrained problems to benchmark the optimization algorithms using for example the following commands:
```
./bench/benchmark_solvers --min-dims 10 --max-dims 100 --convex --epsilon 1e-6 --iterations 1000
```

The following (line-search based) optimization methods are built-in:
```
./apps/info --solver
|----------|------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| solvers  | description                                    | configuration                                                                                                                                                                                      |
|----------|------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| bfgs     | quasi-newton method (BFGS)                     | {"c1":"0.0001","c2":"0.9","init":"quadratic","inits":"[unit,quadratic,consistent]","strat":"interpolate","strats":"[back-armijo,back-wolfe,back-swolfe,interpolate,cg-descent]"}                   |
| broyden  | quasi-newton method (Broyden)                  | {"c1":"0.0001","c2":"0.9","init":"quadratic","inits":"[unit,quadratic,consistent]","strat":"interpolate","strats":"[back-armijo,back-wolfe,back-swolfe,interpolate,cg-descent]"}                   |
| cgd      | nonlinear conjugate gradient descent (default) | {"c1":"0.0001","c2":"0.1","init":"quadratic","inits":"[unit,quadratic,consistent]","orthotest":"0.1","strat":"interpolate","strats":"[back-armijo,back-wolfe,back-swolfe,interpolate,cg-descent]"} |
| cgd-cd   | nonlinear conjugate gradient descent (CD)      | {"c1":"0.0001","c2":"0.1","init":"quadratic","inits":"[unit,quadratic,consistent]","orthotest":"0.1","strat":"interpolate","strats":"[back-armijo,back-wolfe,back-swolfe,interpolate,cg-descent]"} |
| cgd-dy   | nonlinear conjugate gradient descent (DY)      | {"c1":"0.0001","c2":"0.1","init":"quadratic","inits":"[unit,quadratic,consistent]","orthotest":"0.1","strat":"interpolate","strats":"[back-armijo,back-wolfe,back-swolfe,interpolate,cg-descent]"} |
| cgd-dycd | nonlinear conjugate gradient descent (DYCD)    | {"c1":"0.0001","c2":"0.1","init":"quadratic","inits":"[unit,quadratic,consistent]","orthotest":"0.1","strat":"interpolate","strats":"[back-armijo,back-wolfe,back-swolfe,interpolate,cg-descent]"} |
| cgd-dyhs | nonlinear conjugate gradient descent (DYHS)    | {"c1":"0.0001","c2":"0.1","init":"quadratic","inits":"[unit,quadratic,consistent]","orthotest":"0.1","strat":"interpolate","strats":"[back-armijo,back-wolfe,back-swolfe,interpolate,cg-descent]"} |
| cgd-fr   | nonlinear conjugate gradient descent (FR)      | {"c1":"0.0001","c2":"0.1","init":"quadratic","inits":"[unit,quadratic,consistent]","orthotest":"0.1","strat":"interpolate","strats":"[back-armijo,back-wolfe,back-swolfe,interpolate,cg-descent]"} |
| cgd-hs   | nonlinear conjugate gradient descent (HS)      | {"c1":"0.0001","c2":"0.1","init":"quadratic","inits":"[unit,quadratic,consistent]","orthotest":"0.1","strat":"interpolate","strats":"[back-armijo,back-wolfe,back-swolfe,interpolate,cg-descent]"} |
| cgd-ls   | nonlinear conjugate gradient descent (LS)      | {"c1":"0.0001","c2":"0.1","init":"quadratic","inits":"[unit,quadratic,consistent]","orthotest":"0.1","strat":"interpolate","strats":"[back-armijo,back-wolfe,back-swolfe,interpolate,cg-descent]"} |
| cgd-n    | nonlinear conjugate gradient descent (N)       | {"c1":"0.0001","c2":"0.1","init":"quadratic","inits":"[unit,quadratic,consistent]","orthotest":"0.1","strat":"interpolate","strats":"[back-armijo,back-wolfe,back-swolfe,interpolate,cg-descent]"} |
| cgd-prp  | nonlinear conjugate gradient descent (PRP+)    | {"c1":"0.0001","c2":"0.1","init":"quadratic","inits":"[unit,quadratic,consistent]","orthotest":"0.1","strat":"interpolate","strats":"[back-armijo,back-wolfe,back-swolfe,interpolate,cg-descent]"} |
| dfp      | quasi-newton method (DFP)                      | {"c1":"0.0001","c2":"0.9","init":"quadratic","inits":"[unit,quadratic,consistent]","strat":"interpolate","strats":"[back-armijo,back-wolfe,back-swolfe,interpolate,cg-descent]"}                   |
| gd       | gradient descent                               | {"c1":"0.1","c2":"0.9","init":"quadratic","inits":"[unit,quadratic,consistent]","strat":"cg-descent","strats":"[back-armijo,back-wolfe,back-swolfe,interpolate,cg-descent]"}                       |
| lbfgs    | limited-memory BFGS                            | {"c1":"0.0001","c2":"0.9","history":"6","init":"quadratic","inits":"[unit,quadratic,consistent]","strat":"interpolate","strats":"[back-armijo,back-wolfe,back-swolfe,interpolate,cg-descent]"}     |
| sr1      | quasi-newton method (SR1)                      | {"c1":"0.0001","c2":"0.9","init":"quadratic","inits":"[unit,quadratic,consistent]","strat":"interpolate","strats":"[back-armijo,back-wolfe,back-swolfe,interpolate,cg-descent]"}                   |
|----------|------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
```

##### Machine learning

A **task** describes a classification or regression problem consisting of separate training and test samples (e.g. image patches) with associated target outputs if any. The library has built-in support for various standard benchmark datasets which are loaded directly from the original (compressed) files.
```
./apps/info --task
|---------------|--------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------|
| task          | description                                            | configuration                                                                                                   |
|---------------|--------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------|
| cifar10       | CIFAR-10 (3x32x32 object classification)               | {"dir":"/Users/cosmin/experiments/databases/cifar10"}                                                           |
| cifar100      | CIFAR-100 (3x32x32 object classification)              | {"dir":"/Users/cosmin/experiments/databases/cifar100"}                                                          |
| fashion-mnist | Fashion-MNIST (1x28x28 fashion article classification) | {"dir":"/Users/cosmin/experiments/databases/fashion-mnist"}                                                     |
| iris          | IRIS (iris flower classification)                      | {"dir":"/Users/cosmin/experiments/databases/iris"}                                                              |
| mnist         | MNIST (1x28x28 digit classification)                   | {"dir":"/Users/cosmin/experiments/databases/mnist"}                                                             |
| svhn          | SVHN (3x32x32 digit classification in the wild)        | {"dir":"/Users/cosmin/experiments/databases/svhn"}                                                              |
| synth-affine  | synthetic noisy affine transformations                 | {"isize":32,"osize":32,"noise":0.001000,"count":1024}                                                           |
| synth-nparity | synthetic n-parity task (classification)               | {"n":32,"count":1024}                                                                                           |
| synth-peak2d  | synthetic peaks in noisy images                        | {"irows":32,"icols":32,"noise":0.001000,"count":1024,"type":"regression","types":"[regression,classification]"} |
| wine          | WINE (wine classification)                             | {"dir":"/Users/cosmin/experiments/databases/wine"}                                                              |
|---------------|--------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------|
```

The standard benchmark datasets can be download to $HOME/experiments/databases using:
```
bash scripts/download_tasks.sh --iris --wine --mnist --fashion-mnist --cifar10 --cifar100 --svhn
```

The image samples can be saved to disk using for example:
```
./apps/info_task --task mnist --task-params dir=$HOME/experiments/databases/mnist --save-dir ./
```

A **model** predicts the correct output for a given input (e.g. an image patch), either its label (if a classification task) or a score (if a regression task). A model is implemented as an acyclic graph of computation nodes also called layers. The following layers are builtin:
```
./apps/info --layer
|-----------|-------------------------------------------------|---------------------------------------------------------------------------|
| layer     | description                                     | configuration                                                             |
|-----------|-------------------------------------------------|---------------------------------------------------------------------------|
| act-pwave | activation: a(x) = x / (1 + x^2)                | null                                                                      |
| act-sigm  | activation: a(x) = exp(x) / (1 + exp(x))        | null                                                                      |
| act-sin   | activation: a(x) = sin(x)                       | null                                                                      |
| act-snorm | activation: a(x) = x / sqrt(1 + x^2)            | null                                                                      |
| act-splus | activation: a(x) = log(1 + e^x)                 | null                                                                      |
| act-ssign | activation: a(x) = x / (1 + |x|)                | null                                                                      |
| act-tanh  | activation: a(x) = tanh(x)                      | null                                                                      |
| act-unit  | activation: a(x) = x                            | null                                                                      |
| affine    | transform:  L(x) = A * x + b                    | {"ocols":"0","omaps":"0","orows":"0"}                                     |
| conv3d    | transform:  L(x) = conv3D(x, kernel) + b        | {"kcols":"1","kconn":"1","kdcol":"1","kdrow":"1","krows":"1","omaps":"1"} |
| mix-plus  | combine: sum 4D inputs                          | null                                                                      |
| mix-tcat  | combine: concat 4D inputs across feature planes | null                                                                      |
| norm3d    | transform: zero-mean & unit-variance            | {"norm":"global","norms":"[global,plane]"}                                |
|-----------|-------------------------------------------------|---------------------------------------------------------------------------|
```

A **loss** function assigns a scalar score to the prediction of a model `y` by comparing it with the ground truth target `t` (if provided): the lower the score, the better the prediction. The library uses the {-1, +1} codification of class labels.
```
./apps/info --loss
|---------------|----------------------------------------------------------------------------------|---------------|
| loss          | description                                                                      | configuration |
|---------------|----------------------------------------------------------------------------------|---------------|
| cauchy        | multivariate regression:     l(y, t) = 1/2 * log(1 + (y - t)^2)                  | null          |
| classnll      | single-label classification: l(y, t) = log(y.exp().sum()) + 1/2 * (1 + t).dot(y) | null          |
| m-cauchy      | multi-label classification:  l(y, t) = 1/2 * log(1 + (1 - y*t)^2)                | null          |
| m-exponential | multi-label classification:  l(y, t) = exp(-y*t)                                 | null          |
| m-logistic    | multi-label classification:  l(y, t) = log(1 + exp(-y*t))                        | null          |
| m-square      | multi-label classification:  l(y, t) = 1/2 * (1 - y*t)^2                         | null          |
| s-cauchy      | single-label classification: l(y, t) = 1/2 * log(1 + (1 - y*t)^2)                | null          |
| s-exponential | single-label classification: l(y, t) = exp(-y*t)                                 | null          |
| s-logistic    | single-label classification: l(y, t) = log(1 + exp(-y*t))                        | null          |
| s-square      | single-label classification: l(y, t) = 1/2 * (1 - y*t)^2                         | null          |
| square        | multivariate regression:     l(y, t) = 1/2 * (y - t)^2                           | null          |
|---------------|----------------------------------------------------------------------------------|---------------|
```

A **trainer** optimizes the parameters of a given model to produce the correct outputs for a given task using the cumulated values of a given loss over the training samples as a numerical optimization criteria. All the available trainers tune all their required hyper parameters on a separate validation dataset.
```
./apps/info --trainer
|---------|--------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------|
| trainer | description        | configuration                                                                                                                                                   |
|---------|--------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------|
| batch   | batch trainer      | {"solver":"lbfgs","solvers":"[cgd,cgd-cd,cgd-dy,cgd-dycd,cgd-dyhs,cgd-fr,cgd-hs,cgd-ls,cgd-n,cgd-prp,gd,lbfgs]","epochs":1024,"epsilon":0.000001,"patience":32} |
|---------|--------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------|
```


#### Examples

The library provides various command line programs and utilities. Each program displays its possible arguments with short explanations by running it with `--help`.

Most notably:
* **apps/info** - prints all registered objects with the associated ID and a short description.
* **apps/info_archive** - loads an archive (e.g. .tar, .tgz, .zip) and prints its description (e.g. file names and sizes in bytes).
* **apps/train** - train a model on a given task.
* **apps/evaluate** - test a model on a given task.
* **bench/benchmark_solvers** - benchmark all optimization methods with varying the line-search parameters on standard test functions.
* **bench/benchmark_models** - bechmark speed-wise some typical models on a synthetic task.

The `scripts` directory contains examples on how to train various models on different tasks.
