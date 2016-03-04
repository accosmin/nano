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

printf "%-24s" "cppcheck..."

log="cppcheck.log"
cppcheck --enable=all --inconclusive --force --template '{file}:{line},{severity},{id},{message}' \
        ${includes} ${sources} 2> ${log} 1> /dev/null

printf "\terrors: %4d\n\n" \
        $(wc -l < cppcheck.log)

# check available compilers
for compiler in ${compilers}
do
        # check compilation on release and debug
        for build in ${builds}
        do
                printf "%-24s" "${compiler} (${build})..."

                log="${compiler}_${build}.log"
                bash ${basedir}/build_${build}.sh --compiler ${compiler} > ${log} 2>&1

                printf "\tfatals: %4d\terrors: %4d\twarnings: %4d\n" \
                        $(grep -i fatal: ${log} | wc -l) \
                        $(grep -i error: ${log} | wc -l) \
                        $(grep -i warning: ${log} | wc -l)
        done

        # run unit tests with sanitizers
        for config in ${configs}
        do
                printf "%-24s\n" "${compiler} (${config})"

                crtdir=$(pwd)
                cd ${basedir}/build-debug-${config}/test
                for test in test_*
                do
                        printf "  -%-21s" "${test}..."

                        log="${crtdir}/${compiler}_${config}_${test}.log"
                        ./${test} > $log 2>&1

                        printf "\terrors: %4d\n" \
                                $(grep -E ".+\:.+\: \[.+/.+\]: check \{.+\} failed" ${log} | wc -l)
                done
                cd ${crtdir}
        done
done


