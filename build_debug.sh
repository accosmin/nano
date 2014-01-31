mkdir -p ./build-debug
cd ./build-debug/
rm -rf *
cmake -DCMAKE_BUILD_TYPE=Debug ../
make -j 8
cd ..


