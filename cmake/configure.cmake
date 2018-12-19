include(cmake/utils.cmake)

# setup compiler (gcc or clang)
if(CMAKE_CXX_COMPILER_ID MATCHES GNU OR CMAKE_CXX_COMPILER_ID MATCHES Clang)
        message(STATUS "Compiling with ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION} ...")

        require_cpp14()

        # set flags
        set(CMAKE_CXX_FLAGS                     "-std=c++14 -pedantic")
if(CMAKE_WITH_TUNE_NATIVE)
        set(CMAKE_CXX_FLAGS                     "${CMAKE_CXX_FLAGS} -march=native -mtune=native")
endif()
if(CMAKE_WITH_WERROR)
        set(CMAKE_CXX_FLAGS                     "${CMAKE_CXX_FLAGS} -Werror")
endif()
if(CMAKE_WITH_WSHADOW)
        set(CMAKE_CXX_FLAGS                     "${CMAKE_CXX_FLAGS} -Wshadow")
endif()
if(CMAKE_WITH_TIME_REPORT)
        set(CMAKE_CXX_FLAGS                     "${CMAKE_CXX_FLAGS} -ftime-report")
endif()
        set(CMAKE_CXX_FLAGS                     "${CMAKE_CXX_FLAGS} -Wall -Wextra")
        set(CMAKE_CXX_FLAGS                     "${CMAKE_CXX_FLAGS} -Woverloaded-virtual")
        set(CMAKE_CXX_FLAGS                     "${CMAKE_CXX_FLAGS} -Wreorder")
        set(CMAKE_CXX_FLAGS                     "${CMAKE_CXX_FLAGS} -Wnon-virtual-dtor")
        set(CMAKE_CXX_FLAGS                     "${CMAKE_CXX_FLAGS} -Wcast-align")
        set(CMAKE_CXX_FLAGS                     "${CMAKE_CXX_FLAGS} -Wno-old-style-cast")
        set(CMAKE_CXX_FLAGS                     "${CMAKE_CXX_FLAGS} -fno-common")
        set(CMAKE_CXX_FLAGS                     "${CMAKE_CXX_FLAGS} -fvisibility=hidden")
        set(CMAKE_CXX_FLAGS                     "${CMAKE_CXX_FLAGS} -DEIGEN_MPL2_ONLY -DEIGEN_DONT_PARALLELIZE -DEIGEN_DEFAULT_TO_ROW_MAJOR")

        if_flag("-Wmisleading-indentation" CMAKE_CXX_FLAGS)
        if_flag("-Wno-missing-braces" CMAKE_CXX_FLAGS)
        if_flag("-pthread" CMAKE_CXX_FLAGS)
if(NOT (CMAKE_CXX_COMPILER_ID MATCHES GNU AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS "6.0"))
        if_flag("-Wno-unused-command-line-argument" CMAKE_CXX_FLAGS)
        if_flag("-Wno-unknown-warning-option" CMAKE_CXX_FLAGS)
endif()

        set(CMAKE_CXX_FLAGS_DEBUG               "-O0 -g -fno-omit-frame-pointer")
        set(CMAKE_CXX_FLAGS_RELEASE             "-O3 -DNDEBUG -DEIGEN_NO_DEBUG")
        set(CMAKE_CXX_FLAGS_RELWITHDEBINFO      "-O3 -g -fno-omit-frame-pointer")
        set(CMAKE_CXX_FLAGS_MINSIZEREL          "-Os -DNDEBUG")

        # set libc++
        if(CMAKE_WITH_LIBCPP)
                setup_libcpp()
        endif()

        # set gold linker
        if(CMAKE_WITH_GOLD)
                setup_gold()
        endif()

        # set sanitizers
        if(CMAKE_WITH_ASAN)
                setup_asan()
        elseif(CMAKE_WITH_USAN)
                setup_usan()
        elseif(CMAKE_WITH_LSAN)
                setup_lsan()
        elseif(CMAKE_WITH_MSAN)
                setup_msan()
        elseif(CMAKE_WITH_TSAN)
                setup_tsan()
        endif()

        # set LTO
        if(CMAKE_WITH_LTO)
                setup_lto()
        elseif(CMAKE_WITH_THIN_LTO)
                setup_thin_lto()
        endif()

# setup compiler (unsupported)
else()
        message(WARNING "Compiling with an unsupported compiler ...")
endif()

# set clang-tidy
if(CMAKE_WITH_CLANG_TIDY)
        setup_clang_tidy()
endif()

# set coverage
if(CMAKE_WITH_COVERAGE)
        setup_coverage()
endif()

# debug definition
if(CMAKE_BUILD_TYPE MATCHES "[Dd][Ee][Bb][Uu][Gg]")
        add_definitions(-DNANO_DEBUG)
endif()

# set ccache
if(CMAKE_WITH_CCACHE)
        setup_ccache()
endif()

# debug build by default
if(NOT CMAKE_BUILD_TYPE)
        set(CMAKE_BUILD_TYPE "Debug")
endif()

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
message(STATUS "ASAN                           " "${CMAKE_WITH_ASAN}")
message(STATUS "MSAN                           " "${CMAKE_WITH_MSAN}")
message(STATUS "TSAN                           " "${CMAKE_WITH_TSAN}")
message(STATUS "TESTS                          " "${CMAKE_WITH_TESTS}")
message(STATUS "BENCH                          " "${CMAKE_WITH_BENCH}")
message(STATUS "CCACHE                         " "${CMAKE_WITH_CCACHE}")
message(STATUS "------------------------------------------------------------------------------" "")
