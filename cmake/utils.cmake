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

macro(if_cxx_flag flag)
        set(CMAKE_REQUIRED_FLAGS "${flag}")
        CHECK_CXX_SOURCE_COMPILES("${TEST_PROGRAM}" COMPILER_SUPPORTS_${flag})

        if(COMPILER_SUPPORTS_${flag})
                set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${flag}")
        endif()
endmacro()

function(if_linker_flag flag)
        set(CMAKE_REQUIRED_FLAGS ${flag})
        CHECK_CXX_SOURCE_COMPILES("${TEST_PROGRAM}" COMPILER_SUPPORTS_FLAG)

        if(COMPILER_SUPPORTS_FLAG)
                set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${flag}")
        endif()
endfunction()

# require C++14 support
function(require_cpp14)
        CHECK_CXX_COMPILER_FLAG(-std=c++14 COMPILER_SUPPORTS_CXX14)

        if(NOT COMPILER_SUPPORTS_CXX14)
                message(FATAL_ERROR "The compiler has no C++14 support.")
        endif()
endfunction()

# setup libc++
function(setup_libcpp)
        if_cxx_flag("-stdlib=libc++")
        if_linker_flag("-lc++abi")
        to_parent()
endfunction()

# setup sanitizers compatible with address sanitizer
function(setup_asan)
        if_cxx_flag("-fsanitize=address")
        if_cxx_flag("-fsanitize=undefined")
        if_cxx_flag("-fno-sanitize=vptr")
        if_cxx_flag("-fsanitize=leak")
        if_cxx_flag("-fsanitize=integer")
        if_cxx_flag("-fsanitize=bounds")
        if_cxx_flag("-fsanitize=bounds-strict")
        to_parent()
endfunction()

# setup thread sanitizer
function(setup_tsan)
        if_cxx_flag("-fsanitize=thread")
        to_parent()
endfunction()

# setup memory sanitizer
function(setup_msan)
        if_cxx_flag("-fsanitize=memory")
        if_cxx_flag("-fsanitize-memory-track-origins")
        if_cxx_flag("-fPIE")
        if_linker_flag("-pie")
        to_parent()
endfunction()

# setup gold linker
function(setup_gold)
        if_cxx_flag("-fuse-ld=gold")
        if_linker_flag("-fuse-ld=gold")
        to_parent()
endfunction()

# setup ccache
function(setup_ccache)
        find_program(CCACHE_PROGRAM ccache)
        if(CCACHE_PROGRAM)
                # Support Unix Makefiles and Ninja
                set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE "${CCACHE_PROGRAM}")
        endif()
endfunction()

# create clang-tidy-based target for static analysis
function(setup_clang_tidy)
        find_program(CLANG_TIDY_BIN NAMES clang-tidy-5.0 clang-tidy)
        find_program(RUN_CLANG_TIDY_BIN NAMES run-clang-tidy-5.0.py run-clang-tidy.py HINTS /usr/share/clang/)

        list(APPEND CLANG_TIDY_ARGS
                ${RUN_CLANG_TIDY_BIN} -clang-tidy-binary ${CLANG_TIDY_BIN} -header-filter=.*/${PROJECT_NAME}/.*)

        add_custom_target(
                clang-tidy
                COMMAND ${CLANG_TIDY_ARGS} -checks=*
                COMMENT "running clang tidy (*)")

        add_custom_target(
                clang-tidy-cert
                COMMAND ${CLANG_TIDY_ARGS} -checks=-*,cert*
                COMMENT "running clang tidy (cert)")

        add_custom_target(
                clang-tidy-misc
                COMMAND ${CLANG_TIDY_ARGS} -checks=-*,misc*
                COMMENT "running clang tidy (misc)")

        add_custom_target(
                clang-tidy-bugprone
                COMMAND ${CLANG_TIDY_ARGS} -checks=-*,bugprone*
                COMMENT "running clang tidy (bugprone)")

        add_custom_target(
                clang-tidy-modernize
                COMMAND ${CLANG_TIDY_ARGS} -checks=-*,modernize*
                COMMENT "running clang tidy (modernize)")

        add_custom_target(
                clang-tidy-performance
                COMMAND ${CLANG_TIDY_ARGS} -checks=-*,performance*
                COMMENT "running clang tidy (performance)")

        add_custom_target(
                clang-tidy-readability
                COMMAND ${CLANG_TIDY_ARGS} -checks=-*,readability*
                COMMENT "running clang tidy (readability)")

        add_custom_target(
                clang-tidy-clang-analyzer
                COMMAND ${CLANG_TIDY_ARGS} -checks=-*,clang-analyzer*
                COMMENT "running clang tidy (clang-analyzer)")

        add_custom_target(
                clang-tidy-cppcoreguidelines
                COMMAND ${CLANG_TIDY_ARGS} -checks=-*,cppcoreguidelines*
                COMMENT "running clang tidy (cppcoreguidelines)")
endfunction()

# setup LTO
function(setup_lto)
        # TODO: setup -flto=thin
        set(CMAKE_REQUIRED_FLAGS "-flto")
        CHECK_CXX_SOURCE_COMPILES("${TEST_PROGRAM}" COMPILER_SUPPORTS_LTO)

        set(CMAKE_REQUIRED_FLAGS "-fno-fat-lto-objects")
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

# setup valgrind within CTest
function(setup_valgrind)
        set(MEMORYCHECK_COMMAND_OPTIONS "${MEMORYCHECK_COMMAND_OPTIONS} --leak-check=full")
        set(MEMORYCHECK_COMMAND_OPTIONS "${MEMORYCHECK_COMMAND_OPTIONS} --track-fds=yes")
        set(MEMORYCHECK_COMMAND_OPTIONS "${MEMORYCHECK_COMMAND_OPTIONS} --track-origins=yes")
        set(MEMORYCHECK_COMMAND_OPTIONS "${MEMORYCHECK_COMMAND_OPTIONS} --trace-children=yes")
        set(MEMORYCHECK_COMMAND_OPTIONS "${MEMORYCHECK_COMMAND_OPTIONS} --error-exitcode=1")
endfunction()

# setup coverage
function(setup_coverage)
        if_cxx_flag("-g ")
        if_cxx_flag("-O0")
        if_cxx_flag("-fprofile-arcs")
        if_cxx_flag("-ftest-coverage")
        if_linker_flag("--coverage")
        to_parent()
endfunction()

# function to create an application
function(make_app app libs)
        get_filename_component(app_name ${app} NAME_WE)
        add_executable(${app_name} ${app})
        target_link_libraries(${app_name} ${libs})
endfunction()

# function to create a unit test application
function(make_test test libs)
        get_filename_component(test_name ${test} NAME_WE)
        add_executable(${test_name} ${test})
        target_link_libraries(${test_name} ${libs})
        add_test(${test_name} ${test_name})
endfunction()
