#!/bin/bash

git submodule update --init

cpus=$(($(nproc)+1))
compilers="g++-5 g++-6 g++-7 g++-8 clang-3.8 clang-5.0 clang-6.0"

for compiler in ${compilers}
do
        mkdir -p build && cd build && rm -rf *
        cmake .. -G Ninja -DCMAKE_BUILD_TYPE=Debug -DCMAKE_CXX_COMPILER=${compiler} -DNANO_WITH_WERROR=ON
        ctest -E "test_task_svhn|test_trainer|test_solver"
done
