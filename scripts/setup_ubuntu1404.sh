#!/bin/bash

sudo add-apt-repository -y "ppa:ubuntu-toolchain-r/test"
sudo add-apt-repository -y "ppa:george-edison55/cmake-3.x"

sudo add-apt-repository -y "deb http://apt.llvm.org/trusty/ llvm-toolchain-trusty main"
sudo add-apt-repository -y "deb http://apt.llvm.org/trusty/ llvm-toolchain-trusty-5.0 main"
sudo add-apt-repository -y "deb http://apt.llvm.org/trusty/ llvm-toolchain-trusty-6.0 main"
sudo add-apt-repository -y "deb http://ppa.launchpad.net/ubuntu-toolchain-r/test/ubuntu trusty main"

wget -O - https://apt.llvm.org/llvm-snapshot.gpg.key|sudo apt-key add -
# Fingerprint: 6084 F3CF 814B 57C1 CF12 EFD5 15CF 4D18 AF4F 7421

sudo apt-get update -y
#sudo apt-get dist-upgrade -y
sudo apt-get install -y ninja-build cmake3 cppcheck valgrind gcovr
sudo apt-get install -y g++-4.9 g++-5 g++-6 g++-7 g++-8
sudo apt-get install -y clang-3.8 clang-5.0 clang-6.0 clang-tidy-6.0
sudo apt-get install -y libarchive-dev libdevil-dev libc++-dev libc++abi-dev
