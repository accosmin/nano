#!/bin/bash

spinner()
{
        local pid=$!
        local delay=0.75
        local spinstr='|/-\'
        while [ "$(ps a | awk '{print $1}' | grep $pid)" ]
        do
                local temp=${spinstr#?}
                printf " [%c]  " "$spinstr"
                local spinstr=$temp${spinstr%"$temp"}
                sleep $delay
                printf "\r"
        done
        printf "      "
}

checks="clang-tidy-misc
        clang-tidy-bugprone
        clang-tidy-modernize
        clang-tidy-performance
        clang-tidy-readability
        clang-tidy-clang-analyzer
        clang-tidy-cppcoreguidelines"

warnings=0
for check in $checks
do
        printf "running $check...\n"
        log=${check//-/_}.log
        ninja $check > $log&
        spinner

        cat $log | grep warning: | grep -oE "[^ ]+$" | sort | uniq -c
        printf "\n"

        count=$(cat $log | grep warning: | sort -u | wc -l)
        warnings=$((warnings + $count))
done

if [[ $warnings -gt 0 ]]
then
        printf "failed with $warnings unique warnings!\n\n"
        exit 1
else
        printf "passed.\n"
fi
