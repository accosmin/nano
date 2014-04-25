# NanoCV

This small (nano) library is used as a sandbox for training and testing models, such as neural networks and convolution networks, on various image classification and object detection problems. 

## Concepts

The library is built around several key concepts mapped to C++ object interfaces. Each object instantation is registered with an *ID* and thus it can be selected 
from command line arguments. 

### Task

A task describes a classification or regression problem consisting of separate training and test image patches with associated target outputs if any. 
This concept maps known machine learning and computer vision benchmarks to a common interface. Well-known datasets that are supported: *MNIST*, *CIFAR-10*, *CIFAR-100*, *CMU-FACES*, *STL-10*, *NORB*, *SVHN*.

### Model

A model predicts the correct output for a given image patch. The output can be a label (if a classification task) or a score (if a regression task). Implemented instances:

* **forward network** - a collection of feed-forward connected layers: the output of a layer is the input of the next. Implemented layers: *convolution*, 
*activation* (hyperbolic tangent, unit, signed normalization), *linear* and *pooling* (maximum, maximum absolute).

### Loss 

A loss function assigns a scalar score to the prediction of a model by comparing it with the ground truth target (if provided). 
The lower the score, the better the prediction. Implemented instances: *class-NLL*, *logistic* and *square*.

### Trainer

A trainer otimizes the parameters of a *given model* to produce the correct outputs for a *given task* using the cumulated values of a *given loss* over the training samples as a numerical optimization criteria. Implemented instances: *batch* (using *L-BFGS*, conjugate gradient descent (*CGD*) or gradient descent (*GD*)), *minibatch* and *stochastic*.

## Usage

### Compilation

To compile (and install) CMake, a C++11 compiler, Boost, Eigen, libTIFF, libPNG and libJPEG are required. The library is tested so far only on ArchLinux x64, but the 
code is written to be cross-platform.

The easiest way of compiling is to run the `build.sh` bash script. The test programs and utilities will be found in the created `build` directory.

### Examples

The library provides various command line programs and utilities. Each program displays its possible arguments with short explanations by running it with `--help`.

* **ncv_info** - prints all registered objects with their IDs and short descriptions.

* **ncv_info_task** - loads a task and prints its detailed description.

* **ncv_trainer** - train a model on a given task.

* **ncv_tester** - test a model on a given task.

The `experiments.sh` contains examples on how to train various models on different tasks.




 
