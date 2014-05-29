mkdir -p ./build-debug
cd ./build-debug/
rm -rf *
cmake -DCMAKE_BUILD_TYPE=Debug -DNANOCV_HAVE_OPENCL=OFF -G "Unix Makefiles" ../
make -j 8
cd ..


