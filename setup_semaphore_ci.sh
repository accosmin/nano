sudo apt-get install -qq software-properties-common
sudo add-apt-repository -y ppa:ubuntu-toolchain-r/test
sudo add-apt-repository -y ppa:george-edison55/cmake-3.x
sudo apt-get update -qq
sudo apt-get upgrade -qq
sudo apt-get install -qq cmake ninja-build g++-4.9 g++-5 clang-3.5 clang-3.6
sudo apt-get install -qq libarchive-dev libbz2-dev libdevil-dev libboost-all-dev libeigen3-dev
