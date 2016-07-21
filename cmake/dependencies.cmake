# Zlib
find_package(ZLIB REQUIRED)
include_directories(SYSTEM ${ZLIB_INCLUDE_DIR})

# BZip2
find_package(BZip2 REQUIRED)
include_directories(SYSTEM ${BZIP2_INCLUDE_DIR})

# DevIL
find_package(DevIL REQUIRED)
include_directories(SYSTEM ${IL_INCLUDE_DIR})

# LibArchive
find_package(LibArchive REQUIRED)
include_directories(SYSTEM ${LibArchive_INCLUDE_DIRS})

# Eigen
find_package(Eigen3)
include_directories(${EIGEN3_INCLUDE_DIR})
add_definitions(-DEIGEN_DONT_PARALLELIZE)

# OpenCL
if(NANO_WITH_OPENCL)
        find_package(OpenCL REQUIRED)
        include_directories(SYSTEM ${OpenCL_INCLUDE_DIRS})
        add_definitions(-DNANO_WITH_OPENCL)
endif()

