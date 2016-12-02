### Nano

Nano provides numerical optimization and machine learning utilities. For example it can be used to train models such as multi-layer perceptrons (classical neural networks) and convolution networks.


#### Compilation

Use a C++14 compiler and install Eigen3.3+, LibArchive, Zlib, BZip2 and DevIL.

Nano is tested on Linux ([gcc 4.9+ | clang 3.6+], CMake 3.1+, Ninja or Make) and OSX (AppleClang7+, homebrew, CMake 3.1+, Ninja or Make). The code is written to be cross-platform, so it may work (with minor fixes) on other platforms as well (e.g. Windows/MSVC).

The easiest way to compile (and install) is to run:
```
bash build.sh --build-type release --generator ninja
```
The test programs and utilities will be found in the `build-release` directory.

To build the debugging version with or without address, memory and thread sanitizers (if available) run:
```
bash build.sh --build-type debug [--asan|--msan|--tsan] --generator ninja
```

It is recommended to use libc++ with clang by issuing the following command:
```
bash build.sh --build-type release --compiler clang++-3.8 --libc++ --generator ninja
```

To display the list of available build options invoke:
```
bash build.sh --help
```

#### Structure

This library is built around several key concepts mapped to C++ object interfaces. Each object type is registered with an **ID** and thus it can be selected from command line arguments. Also new objects can be easily registered and then they are automatically visible across the library and its associated programs.

The list of all supported objects and their parameters is available using:
```
./apps/info --help
```

##### Numerical optimization

The **batch optimizer** and the **stochastic optimizer** are gradient-based methods used for minimizing generic multi-dimensional functions. They are suitable for large-scale numerical optimization which are often the product of machine learning problems.

Additionally, Nano provides a large set of unconstrained problems to benchmark the optimization algorithms using for example the following commands:
```
./apps/benchmark_batch --min-dims 10 --max-dims 100 --convex --epsilon 1e-6 --iterations 1000
./apps/benchmark_stoch --min-dims 1 --max-dims 4 --convex --epsilon 1e-4 --epoch-size 100 --epochs 1000
```

Nano has built-in support for the following batch (line-search based) optimization methods:
* gradient descent (gd)
* various non-linear conjugate gradient descent (cgd)
* limited-memory BFGS (lbfgs)

Nano has built-in support for the following stochastic optimization methods:
* stochastic gradient (sg)
* stochastic gradient descent with momentum (sgm)
* normalized gradient descent (ngd)
* Nesterov's accelerated gradient (ag) with or without function value (agfr) or gradient (aggr) restarts
* adaptive methods (adadelta, adagrad, adam)
* stochastic variance reduced gradient (svrg)
* averaged stochastic gradient descent (asgd)


##### Machine learning

A **task** describes a classification or regression problem consisting of separate training and test samples (e.g. image patches) with associated target outputs if any. The library has built-in support for various standard benchmark datasets which are loaded directly from the original (compressed) files.

Nano has built-in support for the following tasks:
* MNIST - digit recognition
* CIFAR-10 - 10-class image classification
* CIFAR-100 - 100-class image classification
* STL-10 - 10-class image classification with additional unsupervised data
* SVHN - digit recognition in the wild
* charset - synthetic character recognition task
* affine - synthetic affine transformation task

The standard benchmark datasets can be download to $HOME/experiments/databases using:
```
bash scripts/download_tasks.sh
```

The image samples can be saved to disk using for example:
```
./apps/info_task --task mnist --task-params dir=$HOME/experiments/databases/mnist --save-dir ./
```

A **model** predicts the correct output for a given image patch, either its label (if a classification task) or a score (if a regression task). The feed-forward models can be constructed by combining various layers like:
* convolution
* activation (hyperbolic tangent, unit, signed normalization, soft plus)
* affine

A **loss** function assigns a scalar score to the prediction of a model by comparing it with the ground truth target (if provided): the lower the score, the better the prediction. The loss functions are combined into training **criteria** to account for all training samples and to regularize the model.

A **trainer** optimizes the parameters of a given model to produce the correct outputs for a given task using the cumulated values of a given loss over the training samples as a numerical optimization criteria. All the available trainers tune all their required hyper parameters on a separate validation dataset.

These configurations can be evaluated on the synthetic *charset* task using for example:
```
./apps/benchmark_trainers --batch --stoch --loss classnll --criterion avg \
  --activation act-snorm --trials 10 --epochs 100 --samples 3000
```

#### Examples

The library provides various command line programs and utilities. Each program displays its possible arguments with short explanations by running it with `--help`.

Most notably:
* **apps/info** - prints all registered objects with the associated ID and a short description.
* **apps/info_task** - loads a task and prints its detailed description.
* **apps/train** - train a model on a given task.
* **apps/evaluate** - test a model on a given task.
* **apps/benchmark_batch** - benchmark all batch optimization methods with varying the line-search parameters on standard test functions.
* **apps/benchmark_stoch** - benchmark all stochastic optimization methods on standard test functions.
* **apps/benchmark_models** - bechmark speed-wise some typical models on a synthetic task.
* **apps/benchmark_trainers** - benchmark all training methods on a synthetic task.

The `scripts` directory contains examples on how to train various models on different tasks.
