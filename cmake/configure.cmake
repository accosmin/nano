# check some compiler flags
include(CheckCXXCompilerFlag)

set(NANO_TEST_PROGRAM "int main() { return 0; }")

CHECK_CXX_COMPILER_FLAG(-std=c++14 COMPILER_SUPPORTS_CXX14)

#CHECK_CXX_COMPILER_FLAG(-fsanitize=address COMPILER_SUPPORTS_SANITIZE_ADDRESS)
#CHECK_CXX_COMPILER_FLAG(-fsanitize=undefined COMPILER_SUPPORTS_SANITIZE_UNDEFINED)
#CHECK_CXX_COMPILER_FLAG(-fsanitize=leak COMPILER_SUPPORTS_SANITIZE_LEAK)
#CHECK_CXX_COMPILER_FLAG(-fsanitize=thread COMPILER_SUPPORTS_SANITIZE_THREAD)

set(CMAKE_REQUIRED_FLAGS "-fsanitize=address")
CHECK_CXX_SOURCE_COMPILES("${NANO_TEST_PROGRAM}" COMPILER_SUPPORTS_SANITIZE_ADDRESS)

set(CMAKE_REQUIRED_FLAGS "-fsanitize=undefined")
CHECK_CXX_SOURCE_COMPILES("${NANO_TEST_PROGRAM}" COMPILER_SUPPORTS_SANITIZE_UNDEFINED)

set(CMAKE_REQUIRED_FLAGS "-fsanitize=vptr")
CHECK_CXX_SOURCE_COMPILES("${NANO_TEST_PROGRAM}" COMPILER_SUPPORTS_SANITIZE_VPTR)

set(CMAKE_REQUIRED_FLAGS "-fsanitize=leak")
CHECK_CXX_SOURCE_COMPILES("${NANO_TEST_PROGRAM}" COMPILER_SUPPORTS_SANITIZE_LEAK)

set(CMAKE_REQUIRED_FLAGS "-fsanitize=integer")
CHECK_CXX_SOURCE_COMPILES("${NANO_TEST_PROGRAM}" COMPILER_SUPPORTS_SANITIZE_INTEGER)

set(CMAKE_REQUIRED_FLAGS "-fsanitize=thread")
CHECK_CXX_SOURCE_COMPILES("${NANO_TEST_PROGRAM}" COMPILER_SUPPORTS_SANITIZE_THREAD)

set(CMAKE_REQUIRED_FLAGS "-fsanitize=memory")
CHECK_CXX_SOURCE_COMPILES("${NANO_TEST_PROGRAM}" COMPILER_SUPPORTS_SANITIZE_MEMORY)

# setup compiler
if(CMAKE_CXX_COMPILER_ID MATCHES GNU OR CMAKE_CXX_COMPILER_ID MATCHES Clang)
        message("++ Compiling with ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION} ...")

        # check C++14
        CHECK_CXX_COMPILER_FLAG(-std=c++14 COMPILER_SUPPORTS_CXX14)

        if(NOT COMPILER_SUPPORTS_CXX14)
                message(FATAL_ERROR "The compiler has no C++14 support.")
        endif()

        # set flags
        set(CMAKE_CXX_FLAGS                     "-std=c++14 -pedantic -march=native -mtune=native")
        set(CMAKE_CXX_FLAGS                     "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wconversion")
        set(CMAKE_CXX_FLAGS                     "${CMAKE_CXX_FLAGS} -Woverloaded-virtual")
        set(CMAKE_CXX_FLAGS                     "${CMAKE_CXX_FLAGS} -Wreorder")
        set(CMAKE_CXX_FLAGS                     "${CMAKE_CXX_FLAGS} -Wsign-promo")
        set(CMAKE_CXX_FLAGS                     "${CMAKE_CXX_FLAGS} -Wsign-conversion")
        set(CMAKE_CXX_FLAGS                     "${CMAKE_CXX_FLAGS} -fno-common")
        set(CMAKE_CXX_FLAGS                     "${CMAKE_CXX_FLAGS} -fvisibility=hidden")
        set(CMAKE_CXX_FLAGS_DEBUG               "-g -fno-omit-frame-pointer")
        set(CMAKE_CXX_FLAGS_RELEASE             "-O3 -DNDEBUG")                         # -DEIGEN_NO_DEBUG")
        set(CMAKE_CXX_FLAGS_RELWITHDEBINFO      "-O3 -g -fno-omit-frame-pointer")       # -DEIGEN_NO_DEBUG")
        set(CMAKE_CXX_FLAGS_MINSIZEREL          "-Os -DNDEBUG")                         # -DEIGEN_NO_DEBUG")
        set(CMAKE_EXE_LINKER_FLAGS              "-flto")

        if(NANO_WITH_LIBCPP)
                set(CMAKE_CXX_FLAGS             "-stdlib=libc++ ${CMAKE_CXX_FLAGS}")
                set(CMAKE_EXE_LINKER_FLAGS      "-lc++abi ${CMAKE_EXE_LINKER_FLAGS}")
        endif()

        if(NOT CMAKE_CXX_COMPILER_ID MATCHES AppleClang)
                set(CMAKE_CXX_FLAGS             "${CMAKE_CXX_FLAGS} -pthread")
        endif()

        # set address sanitizer
        if(NANO_WITH_ASAN)
                if(COMPILER_SUPPORTS_SANITIZE_ADDRESS)
                        set(CMAKE_CXX_FLAGS     "${CMAKE_CXX_FLAGS} -fsanitize=address")
                endif()
                if(COMPILER_SUPPORTS_SANITIZE_UNDEFINED)
                        set(CMAKE_CXX_FLAGS     "${CMAKE_CXX_FLAGS} -fsanitize=undefined")
                        if(COMPILER_SUPPORTS_SANITIZE_VPTR)
                        set(CMAKE_CXX_FLAGS     "${CMAKE_CXX_FLAGS} -fno-sanitize=vptr")
                        endif()
                endif()
                if(COMPILER_SUPPORTS_SANITIZE_LEAK)
                        set(CMAKE_CXX_FLAGS     "${CMAKE_CXX_FLAGS} -fsanitize=leak")
                endif()
                if(COMPILER_SUPPORTS_SANITIZE_INTEGER)
                        set(CMAKE_CXX_FLAGS     "${CMAKE_CXX_FLAGS} -fsanitize=integer")
                endif()

        # set memory sanitizer
        elseif(NANO_WITH_MSAN)
                if(COMPILER_SUPPORTS_SANITIZE_MEMORY)
                        set(CMAKE_CXX_FLAGS     "${CMAKE_CXX_FLAGS} -fsanitize=memory -fsanitize-memory-track-origins -fPIE")
                        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -pie")
                endif()

        # set thread sanitizer
        elseif(NANO_WITH_TSAN)
                if(COMPILER_SUPPORTS_SANITIZE_THREAD)
                        set(CMAKE_CXX_FLAGS     "${CMAKE_CXX_FLAGS} -fsanitize=thread")
                endif()
        endif()

else()
        message(WARNING "Compiling with an unsupported compiler ...")
endif()

# debug
if(CMAKE_BUILD_TYPE MATCHES "[Dd][Ee][Bb][Uu][Gg]")
        add_definitions(-DNANO_DEBUG)
endif()
