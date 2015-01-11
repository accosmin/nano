cmake_minimum_required(VERSION 3.1)

project(nanocv)

######################################################################
# External dependencies
######################################################################

option(NANOCV_WITH_CUDA         "Build with CUDA as dependency"                         OFF)
option(NANOCV_WITH_OPENCL       "Build with OpenCL as dependency"                       OFF)
option(NANOCV_WITH_ASAN         "build with address sanitizer suppport (debug, gcc)"    OFF)
option(NANOCV_WITH_LSAN         "build with leak sanitizer suppport (debug, gcc)"       OFF)
option(NANOCV_WITH_TSAN         "build with thread sanitizer suppport (debug, gcc)"     OFF)

# Zlib & BZip2
find_package(ZLIB REQUIRED)
find_package(BZip2 REQUIRED)

include_directories(${ZLIB_INCLUDE_DIR})
include_directories(${BZIP2_INCLUDE_DIR})

# Boost
include(FindBoost)
set(Boost_USE_MULTITHREADED ON)
find_package(Boost COMPONENTS serialization program_options filesystem system)
include_directories(SYSTEM ${Boost_INCLUDE_DIRS})

# DevIL
find_package(DevIL REQUIRED)
include_directories(${IL_INCLUDE_DIR})

# LibArchive
find_package(LibArchive REQUIRED)
include_directories(${LibArchive_INCLUDE_DIRS})

# OpenCL
if(NANOCV_WITH_OPENCL)
        find_package(OpenCL 1.1 REQUIRED)
        if(OpenCL_FOUND)
                include_directories(${OpenCL_INCLUDE_DIRS})
                add_definitions(-DNANOCV_WITH_OPENCL)
        else(OpenCL_FOUND)
                message(FATAL_ERROR "Cannot find OpenCL libraries!")
        endif()
endif()

# CUDA
if(NANOCV_WITH_CUDA)
        find_package(CUDA QUIET REQUIRED)
        if(CUDA_FOUND)
                set(CUDA_PROPAGATE_HOST_FLAGS OFF)
                set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-O3 -gencode arch=compute_20,code=sm_20)
                add_definitions(-DNANOCV_WITH_CUDA)
        else(CUDA_FOUND)
                message(FATAL_ERROR "Cannot find CUDA libraries!")
        endif()
endif()

# OpenMP
find_package(OpenMP QUIET)
if(OPENMP_FOUND)
	set(NANOCV_OPENMP_FLAG "-fopenmp")
endif()

# Set the compiler flags
#set(CMAKE_C_COMPILER                    "/usr/bin/clang")
#set(CMAKE_CXX_COMPILER                  "/usr/bin/clang++")
#set(CMAKE_CXX_COMPILER                  "g++-4.9")

if("${CMAKE_CXX_COMPILER_ID}" MATCHES "GNU")
	message("Compiling with GCC ...")

	set(CMAKE_CXX_FLAGS			"-Wall -std=c++11 -march=native -mtune=native ${NANOCV_OPENMP_FLAG}")
	set(CMAKE_CXX_FLAGS_DEBUG		"-g -Og -fno-omit-frame-pointer")
	set(CMAKE_CXX_FLAGS_RELEASE             "-O3 -DNDEBUG")         # -DEIGEN_NO_DEBUG")
	set(CMAKE_CXX_FLAGS_RELWITHDEBINFO      "-O2 -g")               # -DEIGEN_NO_DEBUG")
	set(CMAKE_CXX_FLAGS_MINSIZEREL          "-Os -DNDEBUG")         # -DEIGEN_NO_DEBUG")
	set(CMAKE_EXE_LINKER_FLAGS              "-flto")

	if(NANOCV_WITH_ASAN)
        	set(CMAKE_CXX_FLAGS_DEBUG       "${CMAKE_CXX_FLAGS_DEBUG} -fsanitize=address -fsanitize=undefined")
	elseif(NANOCV_WITH_LSAN)
        	set(CMAKE_CXX_FLAGS_DEBUG       "${CMAKE_CXX_FLAGS_DEBUG} -fsanitize=leak")
	elseif(NANOCV_WITH_TSAN)
        	set(CMAKE_CXX_FLAGS_DEBUG       "${CMAKE_CXX_FLAGS_DEBUG} -fsanitize=thread -fPIE")
        	set(CMAKE_EXE_LINKER_FLAGS      "${CMAKE_EXE_LINKER_FLAGS} -pie")
	endif()

elseif("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang")
	message("Compiling with CLANG ...")

	set(CMAKE_CXX_FLAGS			"-Wall -std=c++11 -march=native -mtune=native ${NANOCV_OPENMP_FLAG}")
	set(CMAKE_CXX_FLAGS_DEBUG		"-g -fno-omit-frame-pointer")
	set(CMAKE_CXX_FLAGS_RELEASE             "-O3 -DNDEBUG")         # -DEIGEN_NO_DEBUG")
	set(CMAKE_CXX_FLAGS_RELWITHDEBINFO      "-O2 -g")               # -DEIGEN_NO_DEBUG")
	set(CMAKE_CXX_FLAGS_MINSIZEREL          "-Os -DNDEBUG")         # -DEIGEN_NO_DEBUG")
	set(CMAKE_EXE_LINKER_FLAGS              "-flto")

	if(NANOCV_WITH_ASAN)
		message(FATAL_ERROR "Clang does not support address sanitizer!")		
	elseif(NANOCV_WITH_LSAN)
        	message(FATAL_ERROR "Clang does not support leak sanitizer!")
	elseif(NANOCV_WITH_TSAN)
        	message(FATAL_ERROR "Clang does not support thread sanitizer!")
	endif()

else()
	message(WARNING "Compiling with an unsupported compiler ...")
endif()

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
message("CUDA                          " "${NANOCV_WITH_CUDA}")
message("OPENCL                        " "${NANOCV_WITH_OPENCL}")
message("------------------------------------------------------------------------------" "")

include_directories(./)

######################################################################
# Build the library
######################################################################

file(GLOB nanocv_sources
        src/*.*
        src/file/*.*
        src/util/*.*
        src/tensor/*.*
        src/optimize/*.*
        src/nanocv/*.*
        src/nanocv/criteria/*.*
        src/nanocv/losses/*.*
        src/nanocv/layers/*.*
        src/nanocv/tasks/*.*
        src/nanocv/models/*.*
        src/nanocv/trainers/*.*
        )

if(NANOCV_WITH_OPENCL)
        file(GLOB nanocv_opencl_sources
                src/opencl/*.*
        )
        set(nanocv_sources "${nanocv_sources};${nanocv_opencl_sources}")
endif()

set(libs "${libs};${Boost_SERIALIZATION_LIBRARY}")
set(libs "${libs};${Boost_PROGRAM_OPTIONS_LIBRARY}")
set(libs "${libs};${Boost_SYSTEM_LIBRARY}")
set(libs "${libs};${Boost_FILESYSTEM_LIBRARY}")
set(libs "${libs};${Boost_IOSTREAMS_LIBRARY}")
set(libs "${libs};${ZLIB_LIBRARIES}")
set(libs "${libs};${BZIP2_LIBRARIES}")
set(libs "${libs};${LibArchive_LIBRARIES}")
set(libs "${libs};${IL_LIBRARIES}")
if(NANOCV_WITH_OPENCL)
        set(libs "${libs};${OpenCL_LIBRARIES}")
endif()

add_library(nanocv
        SHARED
        ${nanocv_sources})
target_link_libraries(nanocv ${libs})

if(NANOCV_WITH_CUDA)
        file(GLOB nanocv_cuda_sources
                src/cuda/*.*
        )

        cuda_add_library(nanocv_cuda STATIC ${nanocv_cuda_sources})
        target_link_libraries(nanocv nanocv_cuda)
endif()

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

# Set the executables
set(nanocv_test_programs
        ncv_test_thread_loop
        ncv_test_thread_pool
        ncv_test_optimize
        ncv_test_optimize_benchmark
        ncv_test_image
        ncv_test_color
        ncv_test_loss
        ncv_test_conv2d
        ncv_test_corr2d
        ncv_test_unroll
        ncv_test_gradient
        ncv_test_timing
        ncv_test_mem
        ncv_test_stats
        ncv_test_sampler
        ncv_test_model_io
        ncv_test_convnet_benchmark
        ncv_test_archive)

set(nanocv_programs
        ncv_max_threads
        ncv_info
        ncv_info_task
        ncv_trainer
        ncv_tester
        ncv_generator)

# Build executables
foreach(proj ${nanocv_test_programs})
        add_executable(${proj} test/${proj}.cpp)
        target_link_libraries(${proj} nanocv ${libs})
endforeach()
foreach(proj ${nanocv_programs})
        add_executable(${proj} apps/${proj}.cpp)
        target_link_libraries(${proj} nanocv ${libs})
endforeach()

######################################################################
# Installing
######################################################################

# executables
install(TARGETS
        ${nanocv_programs}
        DESTINATION bin)

#set_target_properties(
#        ${nanocv_programs}
#        PROPERTIES INSTALL_RPATH "../lib:./:./lib")

# libraries
install(TARGETS
        nanocv
        DESTINATION lib)

# headers
install(DIRECTORY ${nanocv_SOURCE_DIR}/src/
        DESTINATION include/nanocv
        FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp")

## http://www.cmake.org/Wiki/CMake_RPATH_handling
#set(CMAKE_SKIP_BUILD_RPATH  FALSE)
#set(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)
#set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")
#set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

#list(FIND CMAKE_PLATFORM_IMPLICIT_LINK_DIRECTORIES "${CMAKE_INSTALL_PREFIX}/lib" isSystemDir)
#if("${isSystemDir}" STREQUAL "-1")
#        set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")
#endif("${isSystemDir}" STREQUAL "-1")
