### Nano

Nano provides numerical optimization and machine learning utilities. For example it can be used to train models such as multi-layer perceptrons (classical neural networks) and convolution networks.

This library is built around several key concepts mapped to C++ object interfaces. Each object type is registered with an **ID** and thus it can be selected from command line arguments. Also new objects can be easily registered and then they are automatically visible across the library and its associated programs.


#### Structure

The **batch optimizer** and the **stochastic optimizer** are gradient-based methods used for minimizing generic multi-dimensional functions. They are suitable for large-scale numerical optimization which are often the product of machine learning problems. Examples of batch optimization methods: `gradient descent`, various `non-linear conjugate gradient descent`, `L-BFGS`. Examples of stochastic optimization methods: `Nesterov's accelerated gradient`, `stochastic gradient` (with or without momentum), `normalized gradient descent`, `ADADELTA`, `ADAGRAD`, `ADAM`. Examples of line-search methods: `backtracking`, `cubic interpolation`, `CG_DESCENT`. Additionally, Nano provides a large set of unconstrained problems to benchmark the optimization algorithms.

A **task** describes a classification or regression problem consisting of separate training and test samples (e.g. image patches) with associated target outputs if any. The library has built-in support for various standard benchmark datasets like: `MNIST`, `CIFAR-10`, `CIFAR-100`, `STL-10`, `SVHN`. These datasets are loaded directly from the original (compressed) files.

A **model** predicts the correct output for a given image patch, either its label (if a classification task) or a score (if a regression task). The feed-forward models can be constructed by combining various layers like: `convolution`, `activation` (hyperbolic tangent, unit, signed normalization) and `affine`.

A **loss** function assigns a scalar score to the prediction of a model by comparing it with the ground truth target (if provided): the lower the score, the better the prediction. The loss functions are combined into training **criteria** to account for all training samples and to regularize the model.

A **trainer** optimizes the parameters of a given model to produce the correct outputs for a given task using the cumulated values of a given loss over the training samples as a numerical optimization criteria. All the available trainers tune all their required hyper parameters on a separate validation dataset. The library provides `batch` and `stochastic` instances.


#### Compilation

Use a C++14 compiler and install Eigen3.2+, LibArchive, Zlib, BZip2 and DevIL.

Nano is tested on Linux ([gcc 4.9+ | clang 3.6+], CMake 3.1+, Ninja or Make) and OSX (AppleClang7+, homebrew, CMake 3.1+, Ninja or Make). It is recommended to use libc++ with clang by issuing the following command `build_release.sh clang++-3.5 --libc++`. The code is written to be cross-platform, so it may work (with minor fixes) on other platforms as well (e.g. Windows/MSVC).

The easiest way to compile (and install) is to run `bash build.sh --build-type release`. The test programs and utilities will be found in the `build-release` directory. To build the debugging version with or without address, memory and thread sanitizers (if available) run `bash build.sh --build-type debug [--asan|--msan|--tsan]`. Invoking the build script with `--help` will display the list of available commands.


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
