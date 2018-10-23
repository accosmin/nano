#!/bin/bash

version=1.85
rm -rf cppcheck-${version}

wget -N https://github.com/danmar/cppcheck/archive/${version}.tar.gz
tar -xvf ${version}.tar.gz

cd cppcheck-${version} && mkdir build && cd build
cmake .. -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/tmp/cppcheck
ninja
ninja install
cd ../../

/tmp/cppcheck/bin/cppcheck --version
/tmp/cppcheck/bin/cppcheck -j 4 --force --quiet --inline-suppr --enable=all --error-exitcode=1 -I../src/ ../src ../apps
