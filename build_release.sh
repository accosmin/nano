mkdir -p ./build-release
cd ./build-release/
rm -rf *

cmake_params=""
cmake_params=${cmake_params}" -DCMAKE_BUILD_TYPE=Release"
cmake_params=${cmake_params}" -DNANOCV_HAVE_CUDA=ON"
cmake_params=${cmake_params}" -DNANOCV_HAVE_OPENCL=ON"
cmake_params=${cmake_params}" -DCMAKE_INSTALL_PREFIX=`pwd`/../install"
cmake_params=${cmake_params}" -G Ninja"

cmake ${cmake_params} ../

ninja
cd ..


