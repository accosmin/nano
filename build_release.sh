mkdir -p ./build-release
cd ./build-release/
rm -rf *
cmake -DCMAKE_BUILD_TYPE=Release -DNANOCV_HAVE_OPENCL=OFF -G "Ninja" ../
ninja -j 2
cd ..


