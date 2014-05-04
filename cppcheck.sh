#!/bin/bash

dirs=""
dirs=${dirs}" ./src/ ./src/common/ ./src/tensor/ ./src/optimize/ ./src/opencl/"
dirs=${dirs}" ./src/nanocv/"
dirs=${dirs}" ./src/nanocv/layers"
dirs=${dirs}" ./src/nanocv/losses"
dirs=${dirs}" ./src/nanocv/models"
dirs=${dirs}" ./src/nanocv/tasks"
dirs=${dirs}" ./src/nanocv/trainers"
dirs=${dirs}" ./src/nanocv/accumulators"

log="cppcheck.log"

echo "includes: "
echo ${dirs//src/ -I src}

echo "sources: "
echo ${dirs// /\*.cpp }

echo "checking ..."
cppcheck --enable=all ${dirs//src/ -I src} ${dirs// /\*.cpp } > ${log} 2>&1
echo ">>> done, results in ${log}."
