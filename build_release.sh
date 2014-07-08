mkdir -p ./build-release
cd ./build-release/
rm -rf *
cmake -DCMAKE_BUILD_TYPE=Release -DNANOCV_HAVE_OPENCL=ON -G "Unix Makefiles" ../
make -j 8
cd ..


