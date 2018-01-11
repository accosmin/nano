#!/bin/bash

cpus=$(./apps/info --sys-logical-cpus)
cpus=${cpus/*./}

ctest --output-on-failure -j $cpus -E "test_task_iris|test_task_wine|test_task_cifar|test_task_svhn|test_task_mnist|test_task_fashion"
