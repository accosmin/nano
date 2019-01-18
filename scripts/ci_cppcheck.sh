#!/bin/bash

version=1.86
rm -rf cppcheck-${version}

wget -N https://github.com/danmar/cppcheck/archive/${version}.tar.gz
tar -xvf ${version}.tar.gz

cd cppcheck-${version} && mkdir build && cd build
cmake .. -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/tmp/cppcheck
ninja > build.log 2>&1
ninja install
cd ../../

/tmp/cppcheck/bin/cppcheck --version
/tmp/cppcheck/bin/cppcheck \
        --enable=all --quiet --std=c++14 --error-exitcode=1 --inline-suppr --force \
        --template='{file}:{line},{severity},{id},{message}' \
        --suppress=shadowFunction \
        --suppress=shadowVar \
        --suppress=missingIncludeSystem \
        -I ../src -I ../src/core -I ../deps/utest \
        ../src ../apps ../tests
