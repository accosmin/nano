# check some compiler flags
include(CheckCXXCompilerFlag)

#CHECK_CXX_COMPILER_FLAG(-fsanitize=address COMPILER_SUPPORTS_SANITIZE_ADDRESS)
#CHECK_CXX_COMPILER_FLAG(-fsanitize=undefined COMPILER_SUPPORTS_SANITIZE_UNDEFINED)
#CHECK_CXX_COMPILER_FLAG(-fsanitize=leak COMPILER_SUPPORTS_SANITIZE_LEAK)
#CHECK_CXX_COMPILER_FLAG(-fsanitize=thread COMPILER_SUPPORTS_SANITIZE_THREAD)

set(TEST_PROGRAM "int main() { return 0; }")

# require C++14 support
function (require_cpp14)
        CHECK_CXX_COMPILER_FLAG(-std=c++14 COMPILER_SUPPORTS_CXX14)

        if(NOT COMPILER_SUPPORTS_CXX14)
                message(FATAL_ERROR "The compiler has no C++14 support.")
        endif()
endfunction()

# setup sanitizers compatible with address sanitizer
function (setup_asan)
        set(CMAKE_REQUIRED_FLAGS "-fsanitize=address")
        CHECK_CXX_SOURCE_COMPILES("${TEST_PROGRAM}" COMPILER_SUPPORTS_SANITIZE_ADDRESS)

        set(CMAKE_REQUIRED_FLAGS "-fsanitize=undefined")
        CHECK_CXX_SOURCE_COMPILES("${TEST_PROGRAM}" COMPILER_SUPPORTS_SANITIZE_UNDEFINED)

        set(CMAKE_REQUIRED_FLAGS "-fsanitize=vptr")
        CHECK_CXX_SOURCE_COMPILES("${TEST_PROGRAM}" COMPILER_SUPPORTS_SANITIZE_VPTR)

        set(CMAKE_REQUIRED_FLAGS "-fsanitize=leak")
        CHECK_CXX_SOURCE_COMPILES("${TEST_PROGRAM}" COMPILER_SUPPORTS_SANITIZE_LEAK)

        set(CMAKE_REQUIRED_FLAGS "-fsanitize=integer")
        CHECK_CXX_SOURCE_COMPILES("${TEST_PROGRAM}" COMPILER_SUPPORTS_SANITIZE_INTEGER)

        if(COMPILER_SUPPORTS_SANITIZE_ADDRESS)
                set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address" PARENT_SCOPE)
        endif()
        if(COMPILER_SUPPORTS_SANITIZE_UNDEFINED)
                set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=undefined" PARENT_SCOPE)
        endif()
        if(COMPILER_SUPPORTS_SANITIZE_VPTR)
                set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-sanitize=vptr" PARENT_SCOPE)
        endif()
        if(COMPILER_SUPPORTS_SANITIZE_LEAK)
                set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=leak" PARENT_SCOPE)
        endif()
        if(COMPILER_SUPPORTS_SANITIZE_INTEGER)
                set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=integer" PARENT_SCOPE)
        endif()
endfunction()

# setup thread sanitizer
function (setup_tsan)
        set(CMAKE_REQUIRED_FLAGS "-fsanitize=thread")
        CHECK_CXX_SOURCE_COMPILES("${TEST_PROGRAM}" COMPILER_SUPPORTS_SANITIZE_THREAD)

        if(COMPILER_SUPPORTS_SANITIZE_THREAD)
                set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=thread" PARENT_SCOPE)
        endif()
endfunction()

# setup memory sanitizer
function (setup_msan)
        set(CMAKE_REQUIRED_FLAGS "-fsanitize=memory")
        CHECK_CXX_SOURCE_COMPILES("${TEST_PROGRAM}" COMPILER_SUPPORTS_SANITIZE_MEMORY)

        if(COMPILER_SUPPORTS_SANITIZE_MEMORY)
                set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=memory -fsanitize-memory-track-origins -fPIE" PARENT_SCOPE)
                set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -pie" PARENT_SCOPE)
        endif()
endfunction()

# setup gold linker
function (setup_gold)
        set(CMAKE_REQUIRED_FLAGS "-fuse-ld=gold")
        CHECK_CXX_SOURCE_COMPILES("${TEST_PROGRAM}" COMPILER_SUPPORTS_GOLD)

        if(COMPILER_SUPPORTS_GOLD)
                set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fuse-ld=gold -Wl,--disable-new-dtags" PARENT_SCOPE)
                set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fuse-ld=gold -Wl,--disable-new-dtags" PARENT_SCOPE)
        endif()
endfunction()

