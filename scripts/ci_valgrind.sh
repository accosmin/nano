#!/bin/bash

valgrind --version

bash ../scripts/download_tasks.sh --iris --wine --mnist --fashion-mnist --cal-housing
#ctest --output-on-failure -T memcheck -E "test_task_cifar|test_task_svhn"

exitcode=0

utests=$(ls tests/test_* | grep -v test_task_cifar | grep -v test_task_svhn)
for utest in ${utests}
do
        valgrind --leak-check=full --track-origins=yes --error-exitcode=1 ${utest}
        if [[ $? -gt 0 ]]
        then
                exitcode=1
        fi
done

exit ${exitcode}
