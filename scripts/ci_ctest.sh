#!/bin/bash

cpus=$(./apps/info --sys-logical-cpus)
cpus=${cpus/*./}
#cpus=$(nproc)

bash ../scripts/download_tasks.sh --iris --wine --mnist --fashion-mnist --cifar10 --cifar100
ctest --output-on-failure -j $cpus -E "test_task_svhn"
