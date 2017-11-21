include(CheckCXXCompilerFlag)

set(TEST_PROGRAM "int main(int, char* []) { return 0; }")

macro(to_parent)
        set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} PARENT_SCOPE)
        set(CMAKE_EXE_LINKER_FLAGS ${CMAKE_EXE_LINKER_FLAGS} PARENT_SCOPE)
        set(CMAKE_SHARED_LINKER_FLAGS ${CMAKE_SHARED_LINKER_FLAGS} PARENT_SCOPE)
endmacro()

# add a flag if available to the toolchain
macro(if_flag flag output)
        string(REPLACE "-" "_" FLAG1 ${flag})
        string(REPLACE "=" "_" FLAG2 ${FLAG1})
        string(REPLACE "+" "p" FLAG3 ${FLAG2})
        string(TOUPPER ${FLAG3} FLAGZ)

        #CHECK_CXX_COMPILER_FLAG(${flag} COMPILER_SUPPORTS${FLAGZ})
        set(CMAKE_REQUIRED_FLAGS "${flag}")
        CHECK_CXX_SOURCE_COMPILES("${TEST_PROGRAM}" COMPILER_SUPPORTS${FLAGZ})

        if(COMPILER_SUPPORTS${FLAGZ})
                set(${output} "${${output}} ${flag}")
        endif()
endmacro()

# require C++14 support
function(require_cpp14)
        CHECK_CXX_COMPILER_FLAG(-std=c++14 COMPILER_SUPPORTS_CXX14)

        if(NOT COMPILER_SUPPORTS_CXX14)
                message(FATAL_ERROR "The compiler has no C++14 support.")
        endif()
endfunction()

# setup libc++
function(setup_libcpp)
        if_flag("-stdlib=libc++" CMAKE_CXX_FLAGS)
        if_flag("-lc++abi" CMAKE_EXE_LINKER_FLAGS)
        to_parent()
endfunction()

# setup sanitizers compatible with address sanitizer
function(setup_asan)
        if_flag("-fsanitize=address" CMAKE_CXX_FLAGS)
        if_flag("-fsanitize=undefined" CMAKE_CXX_FLAGS)
        if_flag("-fsanitize=leak" CMAKE_CXX_FLAGS)
        if_flag("-fsanitize=integer" CMAKE_CXX_FLAGS)
        if_flag("-fsanitize=nullability" CMAKE_CXX_FLAGS)
        if_flag("-fsanitize=unsigned-integer-overflow" CMAKE_CXX_FLAGS)
        if_flag("-fno-sanitize=vptr" CMAKE_CXX_FLAGS)
        if_flag("-O1" CMAKE_CXX_FLAGS)
        if_flag("-fno-optimize-sibling-calls" CMAKE_CXX_FLAGS)
        to_parent()
endfunction()

# setup thread sanitizer
function(setup_tsan)
        if_flag("-fsanitize=thread" CMAKE_CXX_FLAGS)
        to_parent()
endfunction()

# setup memory sanitizer
function(setup_msan)
        if_flag("-fsanitize=memory" CMAKE_CXX_FLAGS)
        if_flag("-fsanitize-memory-track-origins" CMAKE_CXX_FLAGS)
        if_flag("-fPIE" CMAKE_CXX_FLAGS)
        if_flag("-pie" CMAKE_EXE_LINKER_FLAGS)
        to_parent()
endfunction()

# setup gold linker
function(setup_gold)
        if_flag("-fuse-ld=gold" CMAKE_CXX_FLAGS)
        if_flag("-fuse-ld=gold" CMAKE_EXE_LINKER_FLAGS)
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
        CHECK_CXX_COMPILER_FLAG("-flto" COMPILER_SUPPORTS_LTO)
        CHECK_CXX_COMPILER_FLAG("-fno-fat-lto-objects" COMPILER_SUPPORTS_NO_FAT_LTO_OBJECTS)

        if(COMPILER_SUPPORTS_LTO)
                get_filename_component(CMAKE_LTO_DIR ${CMAKE_CXX_COMPILER} DIRECTORY)
                if(CMAKE_CXX_COMPILER_ID MATCHES GNU)
                        find_program(CMAKE_LTO_AR NAMES ${CMAKE_TOOLCHAIN_PREFIX}gcc-ar ${CMAKE_TOOLCHAIN_SUFFIX} HINTS ${CMAKE_LTO_DIR})
                        find_program(CMAKE_LTO_NM NAMES ${CMAKE_TOOLCHAIN_PREFIX}gcc-nm HINTS ${CMAKE_LTO_DIR})
                        find_program(CMAKE_LTO_RANLIB NAMES ${CMAKE_TOOLCHAIN_PREFIX}gcc-ranlib HINTS ${CMAKE_LTO_DIR})
                elseif(CMAKE_CXX_COMPILER_ID MATCHES Clang)
                        find_program(CMAKE_LTO_AR NAMES ${CMAKE_TOOLCHAIN_PREFIX}llvm-ar ${CMAKE_TOOLCHAIN_SUFFIX} HINTS ${CMAKE_LTO_DIR})
                        find_program(CMAKE_LTO_NM NAMES ${CMAKE_TOOLCHAIN_PREFIX}llvm-nm HINTS ${CMAKE_LTO_DIR})
                        find_program(CMAKE_LTO_RANLIB NAMES ${CMAKE_TOOLCHAIN_PREFIX}llvm-ranlib HINTS ${CMAKE_LTO_DIR})
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
        if_flag("-g" CMAKE_CXX_FLAGS)
        if_flag("-O0" CMAKE_CXX_FLAGS)
        if_flag("-fprofile-arcs" CMAKE_CXX_FLAGS)
        if_flag("-ftest-coverage" CMAKE_CXX_FLAGS)
        if_flag("--coverage" CMAKE_EXE_LINKER_FLAGS)
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
