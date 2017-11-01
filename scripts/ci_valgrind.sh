#!/bin/bash

valgrind --version
ctest --output-on-failure -T memcheck -E "test_task_*"
