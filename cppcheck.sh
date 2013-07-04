#!/bin/bash

dirs="./src/ ./src/core/ ./src/activation/ ./src/loss/ ./src/model/ ./src/task/ "
log="cppcheck.log"

echo "includes: "
echo ${dirs//src/ -I src}

echo "sources: "
echo ${dirs// /\*.cpp }

echo "checking ..."
cppcheck --enable=all ${dirs//src/ -I src} ${dirs// /\*.cpp } > ${log} 2>&1
echo ">>> done, results in ${log}."
