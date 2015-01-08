#!/bin/bash

dirs=""
dirs=${dirs}" ./src/"
dirs=${dirs}" ./src/cuda/"
dirs=${dirs}" ./src/file/"
dirs=${dirs}" ./src/util/"
dirs=${dirs}" ./src/tensor/"
dirs=${dirs}" ./src/optimize/"
dirs=${dirs}" ./src/opencl/"
dirs=${dirs}" ./src/nanocv/"
dirs=${dirs}" ./src/nanocv/criteria/"
dirs=${dirs}" ./src/nanocv/layers/"
dirs=${dirs}" ./src/nanocv/losses/"
dirs=${dirs}" ./src/nanocv/models/"
dirs=${dirs}" ./src/nanocv/tasks/"
dirs=${dirs}" ./src/nanocv/trainers/"

log="cppcheck.log"

echo "includes: "
echo ${dirs//src/ -I src}

echo "sources: "
echo ${dirs// /\*.cpp }
echo ${dirs// /\*.hpp }
echo ${dirs// /\*.h }
echo ${dirs// /\*.cu }

echo "checking ..."
cppcheck --enable=all --force ${dirs//src/ -I src} ${dirs// /\*.cpp /\*.cu /\*.h /\*.hpp} > ${log} 2>&1
echo ">>> done, results in ${log}."
