#!/bin/bash

checks="clang-tidy-bugprone clang-tidy-modernize clang-tidy-performance clang-tidy-clang-analyzer"
extlog="clang_tidy.log"

rm -f $extlog
for check in $checks
do
        printf "running $check ...\n"
        log=${check//-/_}.log
        ninja $check > $log
        warnings=$(grep -E "warning: " $log | wc -l)
        errors=$(grep -E "error: " $log | wc -l)

        printf "\terrors: %3d, warnings: %3d\n\n" $errors $warnings

        #cat $log | grep -v "header-filter" | grep -v "warnings generated" | grep -v "non-user code" >> $extlog
        cat $log >> $extlog
done

if [[ -n $(grep -E "warning: |error: " $extlog) ]]
then
        printf "clang-tidy detected the following warning and errors:\n\n"
        grep --color -E '^|warning: |error: ' $extlog
        exit 1
else
        printf "passed\n"
fi
