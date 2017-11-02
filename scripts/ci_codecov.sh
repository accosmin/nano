#!/bin/bash


cpus=$(./apps/info --sys-logical-cpus)
cpus=${cpus/*./}

ctest --output-on-failure -j $cpus -E "test_task_*"
cd .. && bash <(curl -s https://codecov.io/bash)
