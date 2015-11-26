### NanoCV

This small (nano) library is used for training and testing models, such as neural networks and convolution networks, on various image classification and object detection problems.


#### Core modules

The core modules are independent of each other and are designed to be as generic as possible.

##### Header only modules

* **math** - numerical utilities, batch and stochastic numerical optimization and line-search methods. `Dependencies: Eigen3.`
* **text** - string processing utilities. `No dependencies.`
* **tensor** - vector, matrix and tensor utilities, 2D/3D convolution and correlations. `Dependencies: Eigen3.`

Most notably the **math** module implements the following: 
* batch optimization methods: `gradient descent`, various `non-linear conjugate gradient descent`, `L-BFGS`.
* stochastic optimization methods: `accelerated gradient`, `stochastic (averaging) gradient`, `ADADELTA`, `ADAGRAD`.
* line-search methods: `backtracking`, `More & Thuente`, `CG_DESCENT`.

##### Libraries

* **io** - I/O utilities, wrappers over LibArchive and std::streams. `Dependencies: LibArchive, Zlib and BZip2.`
* **thread** - thread pool, loop processing in parallel. `No dependencies.`


#### Cortex module

The **cortex** module provides basic image I/O and processing and a machine learning interface. `Dependencies: Boost, DevIL.`

This module is built around several key concepts mapped to C++ object interfaces. Each object type is registered with an **ID** and thus it can be selected from command line arguments. Also new objects can be easily registered and then they are automatically visible across the library and its associated programs.

A **task** describes a classification or regression problem consisting of separate training and test image patches with associated target outputs if any. The library has built-in support for various standard benchmark datasets like: `MNIST`, `CIFAR-10`, `CIFAR-100`, `STL-10`, `SVHN`, `NORB`. These datasets are loaded directly from the original (compressed) files.

A **model** predicts the correct output for a given image patch, either its label (if a classification task) or a score (if a regression task). The feed-forward models can be constructed by combining various layers like: `convolution`, `activation` (hyperbolic tangent, unit, signed normalization), `linear` and `pooling`.

A **loss** function assigns a scalar score to the prediction of a model by comparing it with the ground truth target (if provided): the lower the score, the better the prediction. The loss functions are combined into training **criteria** to account for all training samples and to regularize the model.

A **trainer** optimizes the parameters of a given model to produce the correct outputs for a given task using the cumulated values of a given loss over the training samples as a numerical optimization criteria. All the available trainers tune all their required hyper parameters on a separate validation dataset. The library provides `batch`, `minibatch` and `stochastic` instances.


#### Compilation

Use a C++14 compiler and install Boost, Eigen3, LibArchive and DevIL. 

NanoCV is tested on ArchLinux ([gcc 4.9.3+ | gcc 5.2+ | clang 3.5+], CMake 3.1+, Ninja or Make) and OSX (clang, homebrew, CMake 3.1+, Ninja or Make). The code is written to be cross-platform, so it may work (with minor fixes) on other platforms as well (e.g. Windows/MSVC).

The easiest way to compile (and install) is to run the `build_release.sh` bash script. The test programs and utilities will be found in the `build-release` directory. The `build_debug.sh` bash script will build the debugging version with and without address, leak and thread sanitizers (if available).


#### Examples

The library provides various command line programs and utilities. Each program displays its possible arguments with short explanations by running it with `--help`.

* **info** - prints all registered objects with the associated ID and a short description.
* **info_task** - loads a task and prints its detailed description.
* **trainer** - train a model on a given task.
* **tester** - test a model on a given task.
* **generator** - creates input image patches that maximally activate an output unit (e.g. associated to a class label).

The `scripts` directory contains examples on how to train various models on different tasks.




 
