cmake_minimum_required(VERSION 3.1)

enable_language(CXX)

project(nano)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_SOURCE_DIR}/cmake)

include(cmake/utils.cmake)
include(cmake/options.cmake)
include(cmake/configure.cmake)
include(cmake/dependencies.cmake)

set(NANO_MAJOR_VERSION 0)
set(NANO_MINOR_VERSION 4)
exec_program("git" ${CMAKE_CURRENT_SOURCE_DIR} ARGS "rev-parse HEAD" OUTPUT_VARIABLE NANO_GIT_COMMIT_HASH)

configure_file(
        ${CMAKE_MODULE_PATH}/version.h.in
        ${CMAKE_SOURCE_DIR}/src/version.h)

message("++ -----------------------------------------------------------------------------" "")
message("++ SYSTEM:                        " "${CMAKE_SYSTEM_NAME}")
message("++ PROCESSOR:                     " "${CMAKE_HOST_SYSTEM_PROCESSOR}")
message("++ COMPILER:                      " "${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message("++ ------------------------------------------------------------------------------" "")
message("++ CXX FLAGS:                     " "${CMAKE_CXX_FLAGS}")
message("++ CXX DEBUG FLAGS:               " "${CMAKE_CXX_FLAGS_DEBUG}")
message("++ CXX RELEASE FLAGS:             " "${CMAKE_CXX_FLAGS_RELEASE}")
message("++ CXX RELWITHDEBINFO FLAGS:      " "${CMAKE_CXX_FLAGS_RELWITHDEBINFO}")
message("++ CXX MINSIZEREL FLAGS:          " "${CMAKE_CXX_FLAGS_MINSIZEREL}")
message("++ CMAKE_EXE_LINKER_FLAGS:        " "${CMAKE_EXE_LINKER_FLAGS}")
message("++ ------------------------------------------------------------------------------" "")
message("++ BUILD TYPE:                    " "${CMAKE_BUILD_TYPE}")
message("++ ------------------------------------------------------------------------------" "")
message("++ ASAN                           " "${NANO_WITH_ASAN}")
message("++ MSAN                           " "${NANO_WITH_MSAN}")
message("++ TSAN                           " "${NANO_WITH_TSAN}")
message("++ TESTS                          " "${NANO_WITH_TESTS}")
message("++ ------------------------------------------------------------------------------" "")

include_directories(${nano_SOURCE_DIR}/src)
include_directories(${nano_SOURCE_DIR}/src/optim)
include_directories(${nano_SOURCE_DIR}/src/cortex)

add_subdirectory(src)
add_subdirectory(apps)
if(NANO_WITH_TESTS)
        add_subdirectory(test)
endif()
