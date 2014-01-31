mkdir -p ./build-release
cd ./build-release/
rm -rf *
cmake -DCMAKE_BUILD_TYPE=Release ../
make -j 8
cd ..


