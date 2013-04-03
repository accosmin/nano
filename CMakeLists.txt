cmake_minimum_required(VERSION 2.8)

project(nanocv)

######################################################################
# External depdendencies
######################################################################

# Boost
include(FindBoost)
set(Boost_USE_MULTITHREADED ON)
find_package(Boost COMPONENTS thread serialization program_options filesystem system)
include_directories(SYSTEM ${Boost_INCLUDE_DIRS})

# Qt4
set(QT_QMAKE_EXECUTABLE qmake-qt4)
find_package(Qt4 REQUIRED)
include(${QT_USE_FILE})
add_definitions(${QT_DEFINITIONS})

# Set the compiler flags
set(extra_cxx_flags                    "-Wall -std=c++11 -fopenmp")
set(CMAKE_CXX_FLAGS                    "${extra_cxx_flags} ${CMAKE_CXX_FLAGS}")
set(CMAKE_CXX_FLAGS_DEBUG              "${extra_cxx_flags} ${CMAKE_CXX_FLAGS_DEBUG}")
set(CMAKE_CXX_FLAGS_RELEASE            "${extra_cxx_flags} ${CMAKE_CXX_FLAGS_RELEASE}")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO     "${extra_cxx_flags} ${CMAKE_CXX_FLAGS_RELWITHDEBINFO}")
set(CMAKE_CXX_FLAGS_MINSIZEREL         "${extra_cxx_flags} ${CMAKE_CXX_FLAGS_MINSIZEREL}")

#set (CMAKE_BUILD_TYPE RELEASE)
#set (CMAKE_BUILD_TYPE DEBUG)

include_directories(./)

######################################################################
# Build the library
######################################################################

file(GLOB NANOCV_SOURCES
        src/*.h
        src/*.cpp
        src/core/*.h
        src/core/*.cpp
#        src/tasks/ncv_*.h
#        src/tasks/ncv_*.cpp
#        src/losses/ncv_*.h
#        src/losses/ncv_*.cpp
#        src/models/ncv_*.h
#        src/models/ncv_*.cpp
        )

add_library(nanocv
        ${NANOCV_SOURCES})

######################################################################
# Build projects
######################################################################

# Set the include directories and linking libraries
set(NANOCV_INCLUDES
        src/
        src/core/
#        src/tasks/
#        src/losses/
#        src/models/
        )

include_directories(
        ${NANOCV_INCLUDES})

set(libs "nanocv;")
set(libs "${libs};${Boost_THREAD_LIBRARY_RELEASE}")
set(libs "${libs};${Boost_SERIALIZATION_LIBRARY_RELEASE}")
set(libs "${libs};${Boost_PROGRAM_OPTIONS_LIBRARY_RELEASE}")
set(libs "${libs};${Boost_SYSTEM_LIBRARY_RELEASE}")
set(libs "${libs};${Boost_FILESYSTEM_LIBRARY_RELEASE}")
set(libs "${libs};${QT_LIBRARIES}")

# Build test programs
add_executable          (ncv_max_threads        app/app_ncv_max_threads.cpp)
target_link_libraries   (ncv_max_threads        ${libs})

add_executable          (ncv_test_thread_loop   app/app_ncv_test_thread_loop.cpp)
target_link_libraries   (ncv_test_thread_loop   ${libs})

add_executable          (ncv_test_thread_pool   app/app_ncv_test_thread_pool.cpp)
target_link_libraries   (ncv_test_thread_pool   ${libs})

add_executable          (ncv_test_optimize      app/app_ncv_test_optimize.cpp)
target_link_libraries   (ncv_test_optimize      ${libs})

add_executable          (ncv_test_image         app/app_ncv_test_image.cpp)
target_link_libraries   (ncv_test_image         ${libs})

add_executable          (ncv_test_color         app/app_ncv_test_color.cpp)
target_link_libraries   (ncv_test_color         ${libs})

# Build utility programs
#TODO: image loading & saving & scaling & channel extraction

######################################################################
# Debugging
######################################################################

message("------------------------------------------------------------------------------" "")
message("CXX FLAGS:                    " "${CMAKE_CXX_FLAGS}")
message("CXX DEBUG FLAGS:              " "${CMAKE_CXX_FLAGS_DEBUG}")
message("CXX RELEASE FLAGS:            " "${CMAKE_CXX_FLAGS_RELEASE}")
message("CXX RELWITHDEBINFO FLAGS:     " "${CMAKE_CXX_FLAGS_RELWITHDEBINFO}")
message("CXX MINSIZEREL FLAGS:         " "${CMAKE_CXX_FLAGS_MINSIZEREL}")
message("------------------------------------------------------------------------------" "")
message("BUILD TYPE:                   " "${CMAKE_BUILD_TYPE}")
message("------------------------------------------------------------------------------" "")
