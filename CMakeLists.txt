cmake_minimum_required(VERSION 3.1)

project(nano)

enable_language(CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# use Eigen3 from submodule
set(ENV{EIGEN3_ROOT} "${CMAKE_SOURCE_DIR}/deps")
set(ENV{EIGEN3_ROOT_DIR} "${CMAKE_SOURCE_DIR}/deps/")
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/deps/eigen3/cmake")

include(cmake/utils.cmake)
include(cmake/options.cmake)
include(cmake/configure.cmake)
include(cmake/dependencies.cmake)

if(NANO_WITH_CCACHE)
        setup_ccache()
endif()

if(NOT CMAKE_BUILD_TYPE)
        set(CMAKE_BUILD_TYPE "Debug")
endif()

set(NANO_MAJOR_VERSION 0)
set(NANO_MINOR_VERSION 4)
exec_program("git" ${CMAKE_CURRENT_SOURCE_DIR} ARGS "rev-parse HEAD" OUTPUT_VARIABLE NANO_GIT_COMMIT_HASH)

configure_file(
        "${CMAKE_SOURCE_DIR}/cmake/version.h.in"
        "${CMAKE_SOURCE_DIR}/src/version.h")

message(STATUS "-----------------------------------------------------------------------------" "")
message(STATUS "SYSTEM:                        " "${CMAKE_SYSTEM_NAME}")
message(STATUS "PROCESSOR:                     " "${CMAKE_HOST_SYSTEM_PROCESSOR}")
message(STATUS "LINKER:                        " "${CMAKE_LINKER}")
message(STATUS "COMPILER:                      " "${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "------------------------------------------------------------------------------" "")
message(STATUS "CXX FLAGS:                     " "${CMAKE_CXX_FLAGS}")
message(STATUS "CXX DEBUG FLAGS:               " "${CMAKE_CXX_FLAGS_DEBUG}")
message(STATUS "CXX RELEASE FLAGS:             " "${CMAKE_CXX_FLAGS_RELEASE}")
message(STATUS "CXX RELWITHDEBINFO FLAGS:      " "${CMAKE_CXX_FLAGS_RELWITHDEBINFO}")
message(STATUS "CXX MINSIZEREL FLAGS:          " "${CMAKE_CXX_FLAGS_MINSIZEREL}")
message(STATUS "CMAKE_EXE_LINKER_FLAGS:        " "${CMAKE_EXE_LINKER_FLAGS}")
message(STATUS "------------------------------------------------------------------------------" "")
message(STATUS "BUILD TYPE:                    " "${CMAKE_BUILD_TYPE}")
message(STATUS "------------------------------------------------------------------------------" "")
message(STATUS "ASAN                           " "${NANO_WITH_ASAN}")
message(STATUS "MSAN                           " "${NANO_WITH_MSAN}")
message(STATUS "TSAN                           " "${NANO_WITH_TSAN}")
message(STATUS "TESTS                          " "${NANO_WITH_TESTS}")
message(STATUS "BENCH                          " "${NANO_WITH_BENCH}")
message(STATUS "CCACHE                         " "${NANO_WITH_CCACHE}")
message(STATUS "------------------------------------------------------------------------------" "")

include_directories(${CMAKE_SOURCE_DIR}/src)
include_directories(SYSTEM ${CMAKE_SOURCE_DIR}/deps/json/single_include)

add_subdirectory(src)
add_subdirectory(apps)

if(NANO_WITH_TESTS)
        include(CTest)
        enable_testing()
        add_subdirectory(tests)
endif()
