#!/bin/bash

# check arguments
if [[ ($# -lt 1) || ("$1" == "help") || ("$1" == "--help") || ("$1" == "-h") ]]
then
        printf "analyze.sh <compilers to evaluate>\n"
        exit 1
fi

compilers="$@"
configs="asan msan" # tsan"
builds="release debug"

basedir=$(dirname $0)"/../"

# cppcheck
idirs=""
idirs=${idirs}" "${basedir}/src/
idirs=${idirs}" "${basedir}/apps/
idirs=${idirs}" "${basedir}/test/

sources=""
includes=""
for idir in ${idirs}
do
        dirs=$(find ${idir} -type d)
        for dir in ${dirs}
        do
                sources=${sources}" "${dir}
                includes=${includes}" -I "${dir}
        done
done

printf "%-16s" "cppcheck..."

log="cppcheck.log"
cppcheck --enable=all --inconclusive --force --template '{file}:{line},{severity},{id},{message}' \
        ${includes} ${sources} 2> ${log} 1> /dev/null

printf "\t%-10s%4d\n\n" \
        "errors:" $(wc -l < cppcheck.log)

# check available compilers
for compiler in ${compilers}
do
        # check compilation on release and debug
        for build in ${builds}
        do
                printf "%-16s" "${compiler} (${build})..."

                log="${compiler}_${build}.log"
                #bash ${basedir}/build_${build}.sh --compiler ${compiler} > ${log} 2>&1

                printf "\t%-10s%4d\t%-10s%4d\t%-10s%4d\n" \
                        "fatals:" $(grep -i fatal: ${log} | wc -l) \
                        "errors:" $(grep -i error: ${log} | wc -l) \
                        "warnings:" $(grep -i warning: ${log} | wc -l)
        done

        # run unit tests with sanitizers
        for config in ${configs}
        do
                printf "%-16s\n" "${compiler} (${config})"

                crtdir=$(pwd)
                cd ${basedir}/build-debug-${config}/test
                for test in test_*
                do
                        printf "\t%-24s" "${test}..."

                        log="${crtdir}/${compiler}_${config}_${test}.log"
                        ./${test} > $log 2>&1

                        printf "\t%-10s%4d\n" \
                                "errors:" $(grep -E ".+\:.+\: \[.+/.+\]: check \{.+\} failed!" ${log} | wc -l)
                done
                cd ${crtdir}
        done
done


