#!/bin/bash

valgrind --version

bash ../scripts/download_tasks.sh --iris --wine --mnist --fashion-mnist --cal-housing

# NB: not using ctest directly because I cannot pass options to memcheck!
#ctest --output-on-failure -T memcheck -E "test_task_cifar|test_task_svhn"

exitcode=0

utests=$(ls tests/test_* | grep -v test_task_cifar | grep -v test_task_svhn)
for utest in ${utests}
do
        printf "Running %s ...\n" ${utest}
        log=${utest/tests\//}.log
        valgrind --tool=memcheck --leak-check=yes --show-reachable=yes --num-callers=50 --error-exitcode=1 \
                --log-file=${log} ${utest}

        if [[ $? -gt 0 ]]
        then
                cat ${log}
                exitcode=1
        fi
        printf "\n"
done

exit ${exitcode}
