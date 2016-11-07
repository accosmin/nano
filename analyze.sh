#!/bin/bash

# check arguments
if [[ ($# -lt 1) || ("$1" == "help") || ("$1" == "--help") || ("$1" == "-h") ]]
then
        printf "analyze.sh [--asan] [--msan] [--tsan] <compilers to evaluate>\n"
        printf " --asan: run unit tests with address-compatible sanitizers\n"
        printf " --msan: run unit tests with memory sanitizer\n"
        printf " --tsan: run unit tests with thread sanitizer\n"
        exit 1
fi

# read arguments
compilers=""
configs="${configs} --build-type;release"
configs="${configs} --build-type;debug"

while [ "$1" != "" ]
do
        case $1 in
                --asan) configs="${configs} --build-type;debug;--asan"
                        ;;
                --msan) configs="${configs} --build-type;debug;--msan"
                        ;;
                --tsan) configs="${configs} --build-type;debug;--tsan"
                        ;;
                * )     compilers=${compilers}" "$1
                        ;;
        esac
        shift
done

basedir=$(dirname $0)

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

printf "\terrors: %3d\n\n" \
        $(wc -l < cppcheck.log)

# check available compilers
for compiler in ${compilers}
do
        # check compilation for all configurations
        for xconfig in ${configs}
        do
                pconfig=${xconfig//;/ }
                nconfig=${xconfig//;/_}
                nconfig=${nconfig//--/}
                nconfig=${nconfig//-/_}
                nconfig=${nconfig/build_type_/}

                printf "%-24s" "${compiler} (${nconfig})..."

                log="${compiler}_${nconfig}.log"
                bash ${basedir}/build.sh --compiler ${compiler} --float ${pconfig} > ${log} 2>&1

                printf "\tfatals: %3d\terrors: %3d\twarnings: %3d\n" \
                        $(grep -i fatal: ${log} | wc -l) \
                        $(grep -i error: ${log} | wc -l) \
                        $(grep -i warning: ${log} | wc -l)
        done

        # run unit tests for all configurations
        for xconfig in ${configs}
        do
                pconfig=${xconfig//;/ }
                nconfig=${xconfig//;/_}
                nconfig=${nconfig//--/}
                nconfig=${nconfig//-/_}
                nconfig=${nconfig/build_type_/}
                tconfig=${xconfig//;/-}
                tconfig=${tconfig// /}
                tconfig=${tconfig//--/}
                tconfig=${tconfig//build-type-/}

                printf "%-24s\n" "${compiler} (${nconfig}) ${tconfig}"

                crtdir=$(pwd)
                cd ${basedir}/build-${tconfig}/test
                for test in $(ls test_* | grep -v test_mnist | grep -v test_cifar10 | grep -v test_stl10 | grep -v test_svhn)
                do
                        printf "  -%-21s" "${test}..."

                        log="${crtdir}/${compiler}_${nconfig}_${test}.log"
                        ./${test} > $log 2>&1

                        ret=$(grep -E "failed with|no errors detected" ${log} | wc -l)
                        if [ "$ret" == "0" ]
                        then
                                crashed="yes"
                        else
                                crashed="no"
                        fi
                        printf "\tunit test errors: %3d\tsanitizer errors: %3d\tcrashed: %3s\n" \
                                $(grep -E ".+\:.+\: \[.+/.+\]: check \{.+\} failed" ${log} | wc -l) \
                                $(grep error: ${log} | wc -l) \
                                ${crashed}
                done
                cd ${crtdir}
        done
done


