#!/bin/bash

dirs=""
dirs=${dirs}" ./libnanocv/"
dirs=${dirs}" ./libnanocv/cuda/"
dirs=${dirs}" ./libnanocv/file/"
dirs=${dirs}" ./libnanocv/util/"
dirs=${dirs}" ./libnanocv/tensor/"
dirs=${dirs}" ./libnanocv/optimize/"
dirs=${dirs}" ./libnanocv/opencl/"
dirs=${dirs}" ./libnanocv/criteria/"
dirs=${dirs}" ./libnanocv/layers/"
dirs=${dirs}" ./libnanocv/losses/"
dirs=${dirs}" ./libnanocv/models/"
dirs=${dirs}" ./libnanocv/tasks/"
dirs=${dirs}" ./libnanocv/trainers/"

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
