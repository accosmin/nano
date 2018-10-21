# Zlib
find_package(ZLIB REQUIRED)
include_directories(SYSTEM ${ZLIB_INCLUDE_DIR})

# DevIL
if(APPLE)
        set(IL_INCLUDE_DIR "/usr/local/opt/devil/include/")
        set(IL_LIBRARIES "/usr/local/opt/devil/lib/libIL.dylib")
else()
        find_package(DevIL REQUIRED)
endif()
include_directories(SYSTEM ${IL_INCLUDE_DIR})

# LibArchive
if(APPLE)
        set(LibArchive_INCLUDE_DIRS "/usr/local/opt/libarchive/include/")
        set(LibArchive_LIBRARIES "/usr/local/opt/libarchive/lib/libarchive.dylib")
else()
        find_package(LibArchive REQUIRED)
endif()
include_directories(SYSTEM ${LibArchive_INCLUDE_DIRS})

# Eigen
find_package(Eigen3 3.3 REQUIRED)
include_directories(SYSTEM ${EIGEN3_INCLUDE_DIR} ${EIGEN3_INCLUDE_DIR}/../)
