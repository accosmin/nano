#!/bin/bash

valgrind --version

bash ../scripts/download_tasks.sh --iris --wine --mnist --fashion-mnist
ctest --output-on-failure -T memcheck -E "test_task_cifar|test_task_svhn"
