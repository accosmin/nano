#!/bin/bash

compilers="${compilers} --compiler;g++-4.9"
compilers="${compilers} --compiler;g++-5"
compilers="${compilers} --compiler;g++-6"
compilers="${compilers} --compiler;clang++-3.5;--libc++"
compilers="${compilers} --compiler;clang++-3.6;--libc++"
compilers="${compilers} --compiler;clang++-3.7;--libc++"
compilers="${compilers} --compiler;clang++-3.8;--libc++"
compilers="${compilers} --compiler;clang++-3.9;--libc++"

builds="${builds} --build-type;debug"
builds="${builds} --build-type;release"
builds="${builds} --build-type;debug;--asan"
#builds="${builds} --build-type;debug;--msan"
#builds="${builds} --build-type;debug;--tsan"

scalars="--float --double"

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
        # check build types
        for build in ${builds}
        do
                # check scalar types
                for scalar in ${scalars}
                do
                        bdir="build-temp"
                        cdir=$(pwd)

                        params="${compiler} ${build} ${scalar} --build-dir ${bdir}"
                        params=${params//;/ }

                        name=${params}
                        name=${name//;/_}
                        name=${name//-/_}
                        name=${name//__/_}
                        name=${name// /}
                        name=${name//compiler/}
                        name=${name//build_dir/}
                        name=${name//build_temp/}
                        name=${name//build_type/}

                        # check compilation
                        printf "%s\n" "${params}"

                        /usr/bin/time -f "%E" bash ${basedir}/build.sh ${params} > ${name}.log 2>&1

                        printf "  -%-21s\t%-6s\n" "compilation time:" "$(tail -n 1 ${name}.log)"
                        printf "  -%-21s\t%-6s\n" "compilation fatals:" "$(grep -i fatal: ${name}.log | wc -l)"
                        printf "  -%-21s\t%-6s\n" "compilation errors:" "$(grep -i error: ${name}.log | wc -l)"
                        printf "  -%-21s\t%-6s\n" "compilation warnings:" "$(grep -i warning: ${name}.log | wc -l)"

                        # check unit tests
                        cd ${basedir}/${bdir}/test
                        for test in $(ls test_* | grep -v test_mnist | grep -v test_cifar10 | grep -v test_stl10 | grep -v test_svhn)
                        do
                                printf "  -%-21s" "${test}..."

                                log="${cdir}/${name}_${test}.log"
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
                        cd ${cdir}

                        # cleanup
                        rm -rf ${bdir}
                done
        done
done


