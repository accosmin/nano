#!/bin/bash

idirs=$@

sources=""
includes=""
for idir in ${idirs}
do
	dirs=`find ${idir} -type d`
	for dir in ${dirs}
	do 
		sources=${sources}" "${dir}
		includes=${includes}" -I "${dir}
	done
done

echo "sources:"
echo ${sources}
echo 

echo "includes:"
echo ${includes}
echo

echo "checking ..."
log="cppcheck.log"
cppcheck -j 4 --enable=all --inconclusive --force --template '{file}:{line},{severity},{id},{message}' ${includes} ${sources} > ${log} 2>&1 
echo ">>> done, results in ${log}."
