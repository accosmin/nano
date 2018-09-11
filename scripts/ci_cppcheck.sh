#!/bin/bash

wget -N https://github.com/danmar/cppcheck/archive/1.84.tar.gz
tar -xvf 1.84.tar.gz

cd cppcheck-1.84 && mkdir build && cd build
cmake .. -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=g++-7 -DCMAKE_INSTALL_PREFIX=/tmp/cppcheck
ninja
ninja install
cd ../../

/tmp/cppcheck/bin/cppcheck --version
/tmp/cppcheck/bin/cppcheck -j $(nproc) --force --quiet --inline-suppr --enable=all --error-exitcode=1 -I../src/ ../src ../apps
