#!/bin/bash

checks="clang-tidy-misc
        clang-tidy-bugprone
        clang-tidy-modernize
        clang-tidy-performance
        clang-tidy-readability
        clang-tidy-clang-analyzer
        clang-tidy-cppcoreguidelines"

extlog="clang_tidy.log"

for check in $checks
do
        printf "running $check ...\n"
        log=${check//-/_}.log
        ninja $check > $log

        cat $log | grep warning: | grep -oE '[^ ]+$' | sort | uniq -c
        printf "\n"

        #cat $log | grep -v "header-filter" | grep -v "warnings generated" | grep -v "non-user code" >> $extlog
        cat $log >> $extlog
done

exit

if [[ -n $(grep -E "warning: |error: " $extlog) ]]
then
        printf "clang-tidy detected the following warning and errors:\n\n"
        grep --color -E '^|warning: |error: ' $extlog
        exit 1
else
        printf "passed\n"
fi
