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
        printf "      \r"
}

checks="clang-tidy-cert
        clang-tidy-misc
        clang-tidy-bugprone
        clang-tidy-modernize
        clang-tidy-performance
        clang-tidy-readability
        clang-tidy-clang-analyzer
        clang-tidy-cppcoreguidelines"

optional_checks="
        clang-tidy-cert
        clang-tidy-readability
        clang-tidy-cppcoreguidelines"

warnings=0
for check in $checks
do
        if [[ -z $(echo $optional_checks | grep $check) ]]
        then
                printf "running $check (mandatory)...\n"
        else
                printf "running $check (optional)...\n"
        fi
        log=${check//-/_}.log
        ninja $check > $log&
        spinner

        cat $log | grep warning: | grep -oE "[^ ]+$" | sort | uniq -c
        printf "\n"

        count=$(cat $log | grep warning: | sort -u | wc -l)
        if [[ -z $(echo $optional_checks | grep $check) ]]
        then
                warnings=$((warnings + $count))
        fi
done

if [[ $warnings -gt 0 ]]
then
        printf "failed with $warnings warnings from the mandatory checks!\n\n"
        exit 1
else
        printf "passed.\n"
fi
