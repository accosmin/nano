#!/bin/bash

cpus=$(./apps/info --sys-logical-cpus)
cpus=${cpus/*./}

pip3 install urllib3
python3 ../scripts/experiment.py --download-tasks "iris|wine|mnist|fashion-mnist"
ctest --output-on-failure -j $cpus -E "test_task_cifar|test_task_svhn"
