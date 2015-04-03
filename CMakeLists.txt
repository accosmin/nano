cmake_minimum_required(VERSION 3.1)

project(nanocv)

######################################################################
# external dependencies
######################################################################

option(NANOCV_WITH_CUDA         "build with CUDA as dependency"                         OFF)
option(NANOCV_WITH_OPENCL       "build with OpenCL as dependency"                       OFF)
option(NANOCV_WITH_ASAN         "build with address sanitizer suppport (debug, gcc)"    OFF)
option(NANOCV_WITH_LSAN         "build with leak sanitizer suppport (debug, gcc)"       OFF)
option(NANOCV_WITH_TSAN         "build with thread sanitizer suppport (debug, gcc)"     OFF)

option(NANOCV_WITH_FLOAT        "build using C++'s float as the default scalar"         OFF)
option(NANOCV_WITH_DOUBLE       "build using C++'s double as the default scalar"        OFF)
option(NANOCV_WITH_LONG_DOUBLE  "build using C++'s long double as the default scalar"   OFF)

# Zlib & BZip2
find_package(ZLIB REQUIRED)
find_package(BZip2 REQUIRED)

include_directories(SYSTEM ${ZLIB_INCLUDE_DIR})
include_directories(SYSTEM ${BZIP2_INCLUDE_DIR})

# Boost
include(FindBoost)
set(Boost_USE_STATIC_LIBS        OFF)
set(Boost_USE_MULTITHREADED      ON)
set(Boost_USE_STATIC_RUNTIME     OFF)
set(BOOST_ALL_DYN_LINK           ON)
find_package(Boost COMPONENTS serialization program_options filesystem system unit_test_framework REQUIRED)
include_directories(SYSTEM ${Boost_INCLUDE_DIRS})

# DevIL
find_package(DevIL REQUIRED)
include_directories(SYSTEM ${IL_INCLUDE_DIR})

# LibArchive
find_package(LibArchive REQUIRED)
include_directories(SYSTEM ${LibArchive_INCLUDE_DIRS})

# OpenCL
if(NANOCV_WITH_OPENCL)
        find_package(OpenCL 1.1 REQUIRED)
        if(OpenCL_FOUND)
                include_directories(SYSTEM ${OpenCL_INCLUDE_DIRS})
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

# scalar type
if(NANOCV_WITH_FLOAT)
        add_definitions(-DNANOCV_WITH_FLOAT)
endif()
if(NANOCV_WITH_DOUBLE)
        add_definitions(-DNANOCV_WITH_DOUBLE)
endif()
if(NANOCV_WITH_LONG_DOUBLE)
        add_definitions(-DNANOCV_WITH_LONG_DOUBLE)
endif()

#set(CMAKE_C_COMPILER                    "/usr/bin/clang")
#set(CMAKE_CXX_COMPILER                  "/usr/bin/clang++")
#set(CMAKE_CXX_COMPILER                  "g++-4.9")

######################################################################
# compilation & linking flags (GCC)
######################################################################

if("${CMAKE_CXX_COMPILER_ID}" MATCHES "GNU")
        message("Compiling with gcc ${CMAKE_CXX_COMPILER_VERSION} ...")

        set(CMAKE_CXX_FLAGS			"-Wall -std=c++11 -march=native -mtune=native ${NANOCV_OPENMP_FLAG}")
        set(CMAKE_CXX_FLAGS                     "${CMAKE_CXX_FLAGS} -fvisibility=hidden")
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

######################################################################
# compilation & linking flags (CLANG)
######################################################################
	
elseif("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang")
        message("Compiling with Clang ${CMAKE_CXX_COMPILER_VERSION} ...")

        set(CMAKE_CXX_FLAGS			"-Wall -std=c++11 -march=native -mtune=native ${NANOCV_OPENMP_FLAG}")
        set(CMAKE_CXX_FLAGS                     "${CMAKE_CXX_FLAGS} -fvisibility=hidden")
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

# do not use Eigen's MT
add_definitions(-DEIGEN_DONT_PARALLELIZE)

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
message("ASAN                          " "${NANOCV_WITH_ASAN}")
message("LSAN                          " "${NANOCV_WITH_LSAN}")
message("TSAN                          " "${NANOCV_WITH_TSAN}")
message("CUDA                          " "${NANOCV_WITH_CUDA}")
message("OPENCL                        " "${NANOCV_WITH_OPENCL}")
message("------------------------------------------------------------------------------" "")

######################################################################
# build the library, programs & tests
######################################################################

include_directories(./)

add_subdirectory(libnanocv)
add_subdirectory(apps)
add_subdirectory(test)

