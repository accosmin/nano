# NanoCV

This small (nano) library is used as a sandbox for training and testing models, such as neural networks and convolution networks, on various image classification and object detection problems. 

Dependencies: gcc 4.9+, CMake 3.1+, Ninja, Boost, Eigen3, LibArchive and DevIL. 

Platform: (Arch)Linux x64. The code is written to be cross-platform, so it may work (with minor fixes) on other plaforms (e.g. OSX/clang, Windows/MSVC).

## Concepts

The library is built around several key concepts mapped to C++ object interfaces. Each object instantation is registered with an **ID** and thus it can be selected 
from command line arguments.

#### Task

A task describes a classification or regression problem consisting of separate training and test image patches with associated target outputs if any. Supported benchmark 
datasets: **MNIST**, **CIFAR-10**, **CIFAR-100**, **CBCL-FACES**, **STL-10**, **SVHN**, **NORB**. These datasets are loaded directly from the original (compressed) files.

#### Model

A model predicts the correct output for a given image patch, either its label (if a classification task) or a score (if a regression task). The library 
implements various networks with any user-selectable combination of layers: **convolution**, **activation** (hyperbolic tangent, unit, signed 
normalization), **linear** and **pooling**.

#### Loss 

A loss function assigns a scalar score to the prediction of a model by comparing it with the ground truth target (if provided): the lower the score, the better the prediction.

#### Trainer

A trainer optimizes the parameters of a given model to produce the correct outputs for a given task using the cumulated values of a given loss over the training samples as 
a numerical optimization criteria. Implemented instances: **batch** (using **L-BFGS**, conjugate gradient descent - **CGD** or gradient descent - **GD**), **minibatch** 
and **stochastic** (using Nesterov's accelerated gradient - **NAG**, stochastic gradient - **SG**, stochastic iterative averaging - **SIA** or stochastic gradient averaging - **SGA**).

## Usage

### Compilation

The easiest way of compiling (and installation) is to run the `build_release.sh` bash script. The test programs and utilities will be found in the `build-release` directory.

The `build_debug.sh` bash script will build the debugging version with and without address, leak and thread gcc/clang-based sanitizers.

### Examples

The library provides various command line programs and utilities. Each program displays its possible arguments with short explanations by running it with `--help`.

* **ncv_info** - prints all registered objects with their IDs and short descriptions.

* **ncv_info_task** - loads a task and prints its detailed description.

* **ncv_trainer** - train a model on a given task.

* **ncv_tester** - test a model on a given task.

* **ncv_generator** - creates input image patches that maximally activates an output unit (e.g. associated to a class label).

The `scripts` directory contains examples on how to train various models on different tasks.




 
