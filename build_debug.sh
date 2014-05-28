mkdir -p ./build-debug
cd ./build-debug/
rm -rf *
cmake -DCMAKE_BUILD_TYPE=Debug -DNANOCV_HAVE_OPENCL=OFF -G "Ninja" ../
ninja -j 2
cd ..


