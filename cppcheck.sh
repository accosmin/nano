#!/bin/bash

dirs=""
dirs=${dirs}" ./src/ ./src/util/ ./src/tensor/ ./src/optimize/ ./src/thread/ ./src/opencl/"
dirs=${dirs}" ./src/layers/ ./src/losses/ ./src/models/ ./src/tasks/ ./src/trainers/"

log="cppcheck.log"

echo "includes: "
echo ${dirs//src/ -I src}

echo "sources: "
echo ${dirs// /\*.cpp }

echo "checking ..."
cppcheck --enable=all ${dirs//src/ -I src} ${dirs// /\*.cpp } > ${log} 2>&1
echo ">>> done, results in ${log}."
