cmake_minimum_required(VERSION 3.1)

project(nano VERSION 0.4.0)

enable_language(CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# use Eigen3 from submodule
set(ENV{EIGEN3_ROOT} "${CMAKE_SOURCE_DIR}/deps")
set(ENV{EIGEN3_ROOT_DIR} "${CMAKE_SOURCE_DIR}/deps/")
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/deps/eigen3/cmake")

include(deps/cmake/cmake/utils.cmake)
include(deps/cmake/cmake/options.cmake)
include(deps/cmake/cmake/configure.cmake)
include(cmake/dependencies.cmake)

exec_program("git" ${CMAKE_CURRENT_SOURCE_DIR} ARGS "rev-parse HEAD" OUTPUT_VARIABLE PROJECT_GIT_COMMIT_HASH)

configure_file(
        "${CMAKE_SOURCE_DIR}/cmake/version.h.in"
        "${CMAKE_SOURCE_DIR}/src/version.h")

include_directories(${CMAKE_SOURCE_DIR}/src)
include_directories(SYSTEM ${CMAKE_SOURCE_DIR}/deps/utest)
include_directories(SYSTEM ${CMAKE_SOURCE_DIR}/deps/json/single_include)

add_subdirectory(src)
add_subdirectory(apps)

if(CMAKE_WITH_TESTS)
        include(CTest)
        enable_testing()
        # NB: This doesn't seem to have any effect!
        setup_valgrind()
        add_subdirectory(tests)
endif()
