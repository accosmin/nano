mkdir -p ./build-debug
cd ./build-debug/
rm -rf *

cmake_params=""
cmake_params=${cmake_params}" -DCMAKE_BUILD_TYPE=Debug"
cmake_params=${cmake_params}" -DNANOCV_HAVE_OPENCL=OFF"
cmake_params=${cmake_params}" -G Ninja"

cmake ${cmake_params} ../

ninja
cd ..


