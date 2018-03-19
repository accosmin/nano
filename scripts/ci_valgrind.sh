#!/bin/bash

valgrind --version
python3 ../scripts/downloader.py --tasks "iris|wine|mnist|fashion-mnist"
ctest --output-on-failure -T memcheck -E "test_task_cifar|test_task_svhn"
