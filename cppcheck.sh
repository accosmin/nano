cppcheck --enable=all -I src/ -I src/core/ -I src/loss/ -I src/model/ -I src/task/ src/*.cpp src/core/*.cpp src/loss/*.cpp src/model/*.cpp src/task/*.cpp > cppcheck.log 2>&1
