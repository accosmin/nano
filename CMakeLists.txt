cmake_minimum_required(VERSION 2.8)

project(nanocv)

######################################################################
# External dependencies
######################################################################

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules/")

option(NANOCV_HAVE_OPENCL       "Build nanocv with OpenCL as dependency"        OFF)

# Boost
include(FindBoost)
set(Boost_USE_MULTITHREADED ON)
find_package(Boost COMPONENTS serialization program_options filesystem system)
include_directories(SYSTEM ${Boost_INCLUDE_DIRS})

# Image formats
find_package(PNG REQUIRED)
find_package(JPEG REQUIRED)
find_package(TIFF REQUIRED)

# OpenCL
if(NANOCV_HAVE_OPENCL)
        find_package(OpenCL 1.1 REQUIRED)
        include_directories(${OPENCL_INCLUDE_DIRS})
	add_definitions(-DNANOCV_HAVE_OPENCL)
endif()

# Set the compiler flags
set(CMAKE_CXX_FLAGS                     "-Wall -std=c++11 -fno-builtin -march=native -mtune=native")
set(CMAKE_CXX_FLAGS_DEBUG               "-g -Og")               # -fsanitize=address")
set(CMAKE_CXX_FLAGS_RELEASE             "-O4 -DNDEBUG")         # -DEIGEN_NO_DEBUG")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO      "-O2 -g")               # -DEIGEN_NO_DEBUG")
set(CMAKE_CXX_FLAGS_MINSIZEREL          "-Os -DNDEBUG")         # -DEIGEN_NO_DEBUG")
set(CMAKE_EXE_LINKER_FLAGS              "-ffunction-sections -flto")

# Do not use Eigen's MT
add_definitions(-DEIGEN_DONT_PARALLELIZE)

#set (CMAKE_BUILD_TYPE RELEASE)
#set (CMAKE_BUILD_TYPE DEBUG)

message("------------------------------------------------------------------------------" "")
message("CXX FLAGS:                    " "${CMAKE_CXX_FLAGS}")
message("CXX DEBUG FLAGS:              " "${CMAKE_CXX_FLAGS_DEBUG}")
message("CXX RELEASE FLAGS:            " "${CMAKE_CXX_FLAGS_RELEASE}")
message("CXX RELWITHDEBINFO FLAGS:     " "${CMAKE_CXX_FLAGS_RELWITHDEBINFO}")
message("CXX MINSIZEREL FLAGS:         " "${CMAKE_CXX_FLAGS_MINSIZEREL}")
message("CMAKE_EXE_LINKER_FLAGS:       " "${CMAKE_EXE_LINKER_FLAGS}")
message("------------------------------------------------------------------------------" "")
message("BUILD TYPE:                   " "${CMAKE_BUILD_TYPE}")
message("------------------------------------------------------------------------------" "")

include_directories(./)

######################################################################
# Build the library
######################################################################

file(GLOB nanocv_sources
        src/*.*
        src/common/*.*
        src/tensor/*.*
        src/optimize/*.*
        src/nanocv/*.*
        src/nanocv/losses/*.*
        src/nanocv/layers/*.*
        src/nanocv/tasks/*.*
        src/nanocv/models/*.*
        src/nanocv/trainers/*.*
        )

if(NANOCV_HAVE_OPENCL)
	file(GLOB nanocv_opencl_sources
	        src/opencl/*.*
        )
        set(nanocv_sources "${nanocv_sources};${nanocv_opencl_sources}")
endif()

add_library(nanocv
        SHARED
        ${nanocv_sources})

######################################################################
# Build projects
######################################################################

# Set the include directories and linking libraries
set(nanocv_includes
        src/
        src/nanocv
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
if(NANOCV_HAVE_OPENCL)
        set(libs "${libs};${OPENCL_LIBRARIES}")
endif()

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
        ncv_test_network_timing)

if(NANOCV_HAVE_OPENCL)
        set(nanocv_test_programs "${nanocv_test_programs};ncv_test_opencl")
endif()

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
