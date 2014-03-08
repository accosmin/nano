#!/bin/bash

dirs=""
dirs=${dirs}" ./src/ ./src/common/ ./src/tensor/ ./src/optimize/"
dirs=${dirs}" ./src/ncv/ ./src/ncv/layers/ ./src/ncv/losses/ ./src/ncv/models/ ./src/ncv/tasks/ ./src/ncv/trainers/"

log="cppcheck.log"

echo "includes: "
echo ${dirs//src/ -I src}

echo "sources: "
echo ${dirs// /\*.cpp }

echo "checking ..."
cppcheck --enable=all ${dirs//src/ -I src} ${dirs// /\*.cpp } > ${log} 2>&1
echo ">>> done, results in ${log}."
