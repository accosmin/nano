cmake_minimum_required(VERSION 2.8)

project(nanocv)

######################################################################
# External dependencies
######################################################################

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules/")

# Boost
include(FindBoost)
set(Boost_USE_MULTITHREADED ON)
find_package(Boost COMPONENTS thread serialization program_options filesystem system)
include_directories(SYSTEM ${Boost_INCLUDE_DIRS})

# Image formats
find_package(PNG REQUIRED)
find_package(JPEG REQUIRED)
find_package(TIFF REQUIRED)

# OpenCL
find_package(OpenCL 1.0 REQUIRED)
include_directories(${OPENCL_INCLUDE_DIRS})

# Set the compiler flags
set(CMAKE_CXX_FLAGS                     "-Wall -std=c++11")
set(CMAKE_CXX_FLAGS_DEBUG               "-g")                   # -fsanitize=address")
set(CMAKE_CXX_FLAGS_RELEASE             "-O4 -DNDEBUG")         # -DEIGEN_NO_DEBUG")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO      "-O2 -g")               # -DEIGEN_NO_DEBUG")
set(CMAKE_CXX_FLAGS_MINSIZEREL          "-Os -DNDEBUG")         # -DEIGEN_NO_DEBUG")

#set (CMAKE_BUILD_TYPE RELEASE)
#set (CMAKE_BUILD_TYPE DEBUG)

message("------------------------------------------------------------------------------" "")
message("CXX FLAGS:                    " "${CMAKE_CXX_FLAGS}")
message("CXX DEBUG FLAGS:              " "${CMAKE_CXX_FLAGS_DEBUG}")
message("CXX RELEASE FLAGS:            " "${CMAKE_CXX_FLAGS_RELEASE}")
message("CXX RELWITHDEBINFO FLAGS:     " "${CMAKE_CXX_FLAGS_RELWITHDEBINFO}")
message("CXX MINSIZEREL FLAGS:         " "${CMAKE_CXX_FLAGS_MINSIZEREL}")
message("------------------------------------------------------------------------------" "")
message("BUILD TYPE:                   " "${CMAKE_BUILD_TYPE}")
message("------------------------------------------------------------------------------" "")

include_directories(./)

######################################################################
# Build the library
######################################################################

file(GLOB nanocv_sources
        src/*.*
        src/util/*.*
        src/thread/*.*
        src/tensor/*.*
        src/optimize/*.*
        src/losses/*.*
        src/layers/*.*
        src/tasks/*.*
        src/models/*.*
        src/trainers/*.*
        src/opencl/*.*
        )

add_library(nanocv
        SHARED
        ${nanocv_sources})

######################################################################
# Build projects
######################################################################

# Set the include directories and linking libraries
set(nanocv_includes
        src/
        )

include_directories(
        ${nanocv_includes})

set(libs "nanocv;")
set(libs "${libs};${Boost_THREAD_LIBRARY_RELEASE}")
set(libs "${libs};${Boost_SERIALIZATION_LIBRARY_RELEASE}")
set(libs "${libs};${Boost_PROGRAM_OPTIONS_LIBRARY_RELEASE}")
set(libs "${libs};${Boost_SYSTEM_LIBRARY_RELEASE}")
set(libs "${libs};${Boost_FILESYSTEM_LIBRARY_RELEASE}")
set(libs "${libs};${JPEG_LIBRARIES}")
set(libs "${libs};${TIFF_LIBRARIES}")
set(libs "${libs};${PNG_LIBRARIES}")
set(libs "${libs};${OPENCL_LIBRARIES}")

# Set the executables
set(nanocv_test_programs
        ncv_test_thread_loop
        ncv_test_thread_pool
        ncv_test_optimize
        ncv_test_image
        ncv_test_color
        ncv_test_loss
        ncv_test_convolution
        ncv_test_network_gradient
        ncv_test_network_forward
        ncv_test_network_backward
        ncv_test_opencl)

set(nanocv_programs
        ncv_max_threads
        ncv_info
        ncv_info_task
        ncv_trainer
        ncv_tester)

# Build executables
foreach(proj ${nanocv_test_programs})
        add_executable(${proj} test/${proj}.cpp)
        target_link_libraries(${proj} ${libs})
endforeach()
foreach(proj ${nanocv_programs})
        add_executable(${proj} apps/${proj}.cpp)
        target_link_libraries(${proj} ${libs})
endforeach()

######################################################################
# Installing
######################################################################

# executables
install(TARGETS
        ${nanocv_programs}
        DESTINATION bin)

set_target_properties(
        ${nanocv_programs}
        PROPERTIES INSTALL_RPATH "../lib:./")

# libraries
install(TARGETS
        nanocv
        DESTINATION lib)

## http://www.cmake.org/Wiki/CMake_RPATH_handling
#set(CMAKE_SKIP_BUILD_RPATH  FALSE)
#set(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)
#set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")
#set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

#list(FIND CMAKE_PLATFORM_IMPLICIT_LINK_DIRECTORIES "${CMAKE_INSTALL_PREFIX}/lib" isSystemDir)
#if("${isSystemDir}" STREQUAL "-1")
#        set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")
#endif("${isSystemDir}" STREQUAL "-1")
