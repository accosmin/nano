#!/bin/bash

version=1.87
rm -rf cppcheck-${version}

wget -N https://github.com/danmar/cppcheck/archive/${version}.tar.gz
tar -xvf ${version}.tar.gz

cd cppcheck-${version} && mkdir build && cd build
cmake .. -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/tmp/cppcheck
ninja > build.log 2>&1
ninja install
cd ../../

/tmp/cppcheck/bin/cppcheck --version

# NB: the warnings are not fatal (exitcode=0) as they are usually false alarms!
/tmp/cppcheck/bin/cppcheck \
        --project=compile_commands.json \
        --enable=all --quiet --std=c++14 --error-exitcode=0 --inline-suppr --force \
        --template='{file}:{line},{severity},{id},{message}' \
        --suppress=shadowFunction \
        --suppress=shadowVar \
        --suppress=unknownMacro \
        --suppress=missingIncludeSystem \
        ../src ../apps ../tests
