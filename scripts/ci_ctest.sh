#!/bin/bash

cpus=$(./apps/info --sys-logical-cpus)
cpus=${cpus/*./}

python3 ../scripts/downloader.py --tasks "iris|wine|mnist|fashion-mnist|cifar10|cifar100"
ctest --output-on-failure -j $cpus -E "test_task_svhn|test_trainer|test_solver"
