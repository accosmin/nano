# check some compiler flags
include(CheckCXXCompilerFlag)

#CHECK_CXX_COMPILER_FLAG(-fsanitize=address COMPILER_SUPPORTS_SANITIZE_ADDRESS)
#CHECK_CXX_COMPILER_FLAG(-fsanitize=undefined COMPILER_SUPPORTS_SANITIZE_UNDEFINED)
#CHECK_CXX_COMPILER_FLAG(-fsanitize=leak COMPILER_SUPPORTS_SANITIZE_LEAK)
#CHECK_CXX_COMPILER_FLAG(-fsanitize=thread COMPILER_SUPPORTS_SANITIZE_THREAD)

set(TEST_PROGRAM "int main(int, char* []) { return 0; }")

macro(to_parent)
        set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} PARENT_SCOPE)
        set(CMAKE_EXE_LINKER_FLAGS ${CMAKE_EXE_LINKER_FLAGS} PARENT_SCOPE)
        set(CMAKE_SHARED_LINKER_FLAGS ${CMAKE_SHARED_LINKER_FLAGS} PARENT_SCOPE)
endmacro()

# require C++14 support
function(require_cpp14)
        CHECK_CXX_COMPILER_FLAG(-std=c++14 COMPILER_SUPPORTS_CXX14)

        if(NOT COMPILER_SUPPORTS_CXX14)
                message(FATAL_ERROR "The compiler has no C++14 support.")
        endif()
endfunction()

# setup sanitizers compatible with address sanitizer
function(setup_asan)
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
                set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address")
        endif()
        if(COMPILER_SUPPORTS_SANITIZE_UNDEFINED)
                set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=undefined")
        endif()
        if(COMPILER_SUPPORTS_SANITIZE_VPTR)
                set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-sanitize=vptr")
        endif()
        if(COMPILER_SUPPORTS_SANITIZE_LEAK)
                set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=leak")
        endif()
        if(COMPILER_SUPPORTS_SANITIZE_INTEGER)
                set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=integer")
        endif()

        to_parent()
endfunction()

# setup thread sanitizer
function(setup_tsan)
        set(CMAKE_REQUIRED_FLAGS "-fsanitize=thread")
        CHECK_CXX_SOURCE_COMPILES("${TEST_PROGRAM}" COMPILER_SUPPORTS_SANITIZE_THREAD)

        if(COMPILER_SUPPORTS_SANITIZE_THREAD)
                set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=thread")
        endif()

        to_parent()
endfunction()

# setup memory sanitizer
function(setup_msan)
        set(CMAKE_REQUIRED_FLAGS "-fsanitize=memory")
        CHECK_CXX_SOURCE_COMPILES("${TEST_PROGRAM}" COMPILER_SUPPORTS_SANITIZE_MEMORY)

        if(COMPILER_SUPPORTS_SANITIZE_MEMORY)
                set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=memory -fsanitize-memory-track-origins -fPIE")
                set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -pie")
        endif()

        to_parent()
endfunction()

# setup gold linker
function(setup_gold)
        set(CMAKE_REQUIRED_FLAGS "-fuse-ld=gold")
        CHECK_CXX_SOURCE_COMPILES("${TEST_PROGRAM}" COMPILER_SUPPORTS_GOLD)

        if(COMPILER_SUPPORTS_GOLD)
                set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fuse-ld=gold")
                set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fuse-ld=gold")
        endif()

        to_parent()
endfunction()

# setup LTO
function(setup_lto)
        set(CMAKE_REQUIRE_FLAGS "-flto")
        CHECK_CXX_SOURCE_COMPILES("${TEST_PROGRAM}" COMPILER_SUPPORTS_LTO)

        set(CMAKE_REQUIRE_FLAGS "-fno-fat-lto-objects")
        CHECK_CXX_SOURCE_COMPILES("${TEST_PROGRAM}" COMPILER_SUPPORTS_NO_FAT_LTO_OBJECTS)

        if(COMPILER_SUPPORTS_LTO)
                get_filename_component(CMAKE_LTO_DIR ${CMAKE_CXX_COMPILER} DIRECTORY)
                if(CMAKE_CXX_COMPILER_ID MATCHES GNU)
                        find_program(CMAKE_LTO_AR NAMES ${_CMAKE_TOOLCHAIN_PREFIX}gcc-ar ${_CMAKE_TOOLCHAIN_SUFFIX} HINTS ${CMAKE_LTO_DIR})
                        find_program(CMAKE_LTO_NM NAMES ${_CMAKE_TOOLCHAIN_PREFIX}gcc-nm HINTS ${CMAKE_LTO_DIR})
                        find_program(CMAKE_LTO_RANLIB NAMES ${_CMAKE_TOOLCHAIN_PREFIX}gcc-ranlib HINTS ${CMAKE_LTO_DIR})
                elseif(CMAKE_CXX_COMPILER_ID MATCHES Clang)
                        find_program(CMAKE_LTO_AR NAMES ${_CMAKE_TOOLCHAIN_PREFIX}llvm-ar ${_CMAKE_TOOLCHAIN_SUFFIX} HINTS ${CMAKE_LTO_DIR})
                        find_program(CMAKE_LTO_NM NAMES ${_CMAKE_TOOLCHAIN_PREFIX}llvm-nm HINTS ${CMAKE_LTO_DIR})
                        find_program(CMAKE_LTO_RANLIB NAMES ${_CMAKE_TOOLCHAIN_PREFIX}llvm-ranlib HINTS ${CMAKE_LTO_DIR})
                endif()

                message("++ CMAKE_AR: ${CMAKE_LTO_AR}")
                message("++ CMAKE_NM: ${CMAKE_LTO_NM}")
                message("++ CMAKE_RANLIB: ${CMAKE_LTO_RANLIB}")

                if(CMAKE_LTO_AR AND CMAKE_LTO_NM AND CMAKE_LTO_RANLIB)
                        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -flto")
                        if(COMPILER_SUPPORTS_NO_FAT_LTO_OBJECTS)
                                set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-fat-lto-objects")
                        endif()
                        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -flto=4")

                        set(CMAKE_AR "${CMAKE_LTO_AR}" PARENT_SCOPE)
                        set(CMAKE_NM "${CMAKE_LTO_NM}" PARENT_SCOPE)
                        set(CMAKE_RANLIB "${CMAKE_LTO_RANLIB}" PARENT_SCOPE)

                        to_parent()
                else()
                        message(WARNING "++ Compiler has LTO support, but binutils wrappers could not be found. Disabling LTO." )
                endif()
        endif()
endfunction()

