mkdir -p ./build-release
cd ./build-release/
rm -rf *

cmake_params=""
cmake_params=${cmake_params}" -DCMAKE_BUILD_TYPE=Release"
cmake_params=${cmake_params}" -DNANOCV_HAVE_OPENCL=OFF"
cmake_params=${cmake_params}" -G Ninja"

cmake ${cmake_params} ../

ninja -j 8
cd ..


