#!/bin/bash

dirs="./src/ ./src/core/ ./src/core/math/ ./src/core/tensor ./src/core/optimize ./src/core/thread ./src/layers/ ./src/losses/ ./src/models/ ./src/tasks/ ./src/trainers/ "
log="cppcheck.log"

echo "includes: "
echo ${dirs//src/ -I src}

echo "sources: "
echo ${dirs// /\*.cpp }

echo "checking ..."
cppcheck --enable=all ${dirs//src/ -I src} ${dirs// /\*.cpp } > ${log} 2>&1
echo ">>> done, results in ${log}."
