#!/bin/bash

build_type=Debug

bash build.sh --build-dir ./build-debug --build-type ${build_type} --asan OFF --lsan OFF --tsan OFF

# does not work with clang!
#bash build.sh --build-dir ./build-debug-asan --build-type ${build_type} --asan ON --lsan OFF --tsan OFF
#bash build.sh --build-dir ./build-debug-lsan --build-type ${build_type} --asan OFF --lsan ON --tsan OFF
#bash build.sh --build-dir ./build-debug-tsan --build-type ${build_type} --asan OFF --lsan OFF --tsan ON

