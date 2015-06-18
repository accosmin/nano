#!/bin/bash

build_type=Debug
build_sys=ninja

bash build.sh --build-dir ./build-debug --build-type ${build_type} --build-sys ${build_sys} --asan OFF --tsan OFF

bash build.sh --build-dir ./build-debug-asan --build-type ${build_type} --build-sys ${build_sys} --asan ON --tsan OFF
bash build.sh --build-dir ./build-debug-tsan --build-type ${build_type} --build-sys ${build_sys} --asan OFF --tsan ON

