# NanoCV

This small (nano) library is used as a sandbox for implementing deep-models, such as neural networks and convolution networks, and testing them on various image 
classification and object detection problems. 

## Concepts

The library is built around several key concepts mapped to C++ object interfaces. Each object instantation is registered with an *ID* and thus it can be selected 
from command line arguments. 

### Task

A task describes a classification or regression problem consisting of separate training and test image patches with associated target outputs if any. 
This concept maps known machine learning and computer vision benchmarks to a common interface. Implemented instances: 

* **MNIST** - digit classification, 28x28 grayscale inputs,

* **CIFAR-10** - 10-class object classification, 32x32 RGB inputs,

* **CMU-FACES** - face detection (binary classification), 19x19 grayscale inputs,

* **STL-10** - 10-class object classification, 96x96 RGB inputs.

### Model

A model predicts the correct output for a given image patch. The output can be a label (if a classification task) or a score (if a regression task). Implemented instances:

* **forward network** - a collection of feed-forward connected layers: the output of a layer is the input of the next. Implemented layers: *convolution*, 
*activation* (hyperbolic tangent, unit, signed normalization) and *pooling/compression* (maximum, maximum absolute).

### Loss 

A loss function assigns a scalar score to the prediction of a model by comparing it with the ground truth target (if provided). 
The lower the score, the better the prediction. Implemented instances:

* **class-NLL** - for multi-label classification problems,

* **logistic** - for classification problems,

* **square** - most usefull for regression problems.

### Trainer

A trainer otimizes the parameters of a *given model* to produce the correct outputs for a *given task* using the cumulated values of a *given loss* over 
the training samples as a numerical optimization criteria. Implemented instances:

* **batch** - a single iteration typically consists of a pass through all training samples. There are several options available: *L-BFGS*, conjugate gradient 
descent (*CGD*) and gradient descent (*GD*).

* **stochastic** - an iteration consists of picking a random sample to update the descent direction. There are several options available: *SGD* (stochastic gradient descent) and *ASGD* (average stochastic gradient descent).
The training is split in two phases: the tuning phase estimates the optimum learning rate parameters on a small subset of samples, while the optimization phase uses the optimum parameters to train the model.

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




 
