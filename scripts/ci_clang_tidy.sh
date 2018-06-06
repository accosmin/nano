#!/bin/bash

spinner()
{
        local pid=$!
        local delay=0.75
        local spinstr='|/-\'
        while [ "$(ps a | awk '{print $1}' | grep $pid)" ]
        do
                local temp=${spinstr#?}
                printf " [%c] " "$spinstr"
                local spinstr=$temp${spinstr%"$temp"}
                sleep $delay
                printf "\r"
        done
        printf "      \r"
}

check=$1

# clang-tidy-misc
# clang-tidy-bugprone
# clang-tidy-modernize
# clang-tidy-performance
# clang-tidy-clang-analyzer"
# clang-tidy-cert
# clang-tidy-readability
# clang-tidy-cppcoreguidelines"

printf "running $check ...\n"
log=${check//-/_}.log
ninja $check > $log&
spinner

cat $log | grep warning: | grep -oE "[^ ]+$" | sort | uniq -c
printf "\n"

warnings=$(cat $log | grep warning: | sort -u | wc -l)
#grep warning: $log
if [[ $warnings -gt 0 ]]
then
        cat $log
fi

if [[ $warnings -gt 0 ]]
then
        printf "failed with $warnings warnings!\n\n"
        exit 1
else
        printf "passed.\n"
        exit 0
fi
