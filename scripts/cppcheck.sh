#!/bin/bash

idirs=$@

dirs=""
for idir in ${idirs}
do
	dirs=${dirs}" "`find ${idir} -type d`
done

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
