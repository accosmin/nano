mkdir -p ./build-release
cd ./build-release/
rm -rf *
cmake -DCMAKE_BUILD_TYPE=Release -DNANOCV_HAVE_OPENCL=TRUE ../
make -j 8
cd ..


