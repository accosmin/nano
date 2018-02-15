include(cmake/utils.cmake)

# setup compiler (gcc or clang)
if(CMAKE_CXX_COMPILER_ID MATCHES GNU OR CMAKE_CXX_COMPILER_ID MATCHES Clang)
        message("++ Compiling with ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION} ...")

        require_cpp14()

        # set flags
        set(CMAKE_CXX_FLAGS                     "-std=c++14 -pedantic -ffunction-sections")
if(NANO_WITH_TUNE_NATIVE)
        set(CMAKE_CXX_FLAGS                     "${CMAKE_CXX_FLAGS} -march=native -mtune=native")
endif()
if(NANO_WITH_WERROR)
        set(CMAKE_CXX_FLAGS                     "${CMAKE_CXX_FLAGS} -Werror")
endif()
if(NANO_WITH_TIME_REPORT)
        set(CMAKE_CXX_FLAGS                     "${CMAKE_CXX_FLAGS} -ftime-report")
endif()
        set(CMAKE_CXX_FLAGS                     "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wconversion")
        set(CMAKE_CXX_FLAGS                     "${CMAKE_CXX_FLAGS} -Woverloaded-virtual")
        set(CMAKE_CXX_FLAGS                     "${CMAKE_CXX_FLAGS} -Wreorder")
        set(CMAKE_CXX_FLAGS                     "${CMAKE_CXX_FLAGS} -Wnon-virtual-dtor")
        set(CMAKE_CXX_FLAGS                     "${CMAKE_CXX_FLAGS} -Wcast-align")
        set(CMAKE_CXX_FLAGS                     "${CMAKE_CXX_FLAGS} -Wno-old-style-cast")
        set(CMAKE_CXX_FLAGS                     "${CMAKE_CXX_FLAGS} -Wno-sign-promo")
        set(CMAKE_CXX_FLAGS                     "${CMAKE_CXX_FLAGS} -Wno-sign-conversion")
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

        set(CMAKE_CXX_FLAGS_DEBUG               "-g -fno-omit-frame-pointer")
        set(CMAKE_CXX_FLAGS_RELEASE             "-O3 -DNDEBUG -DEIGEN_NO_DEBUG")
        set(CMAKE_CXX_FLAGS_RELWITHDEBINFO      "-O3 -g -fno-omit-frame-pointer")
        set(CMAKE_CXX_FLAGS_MINSIZEREL          "-Os -DNDEBUG")

        # set libc++
        if(NANO_WITH_LIBCPP)
                setup_libcpp()
        endif()

        # set gold linker
        if(NANO_WITH_GOLD)
                setup_gold()
        endif()

        # set sanitizers
        if(NANO_WITH_ASAN)
                setup_asan()
        elseif(NANO_WITH_MSAN)
                setup_msan()
        elseif(NANO_WITH_TSAN)
                setup_tsan()
        endif()

        # set LTO
        if(NANO_WITH_LTO)
                setup_lto()
        endif()

# setup compiler (unsupported)
else()
        message(WARNING "++ Compiling with an unsupported compiler ...")
endif()

# set clang-tidy
if(NANO_WITH_CLANG_TIDY)
        setup_clang_tidy()
endif()

# set coverage
if(NANO_WITH_COVERAGE)
        setup_coverage()
endif()

# debug
if(CMAKE_BUILD_TYPE MATCHES "[Dd][Ee][Bb][Uu][Gg]")
        add_definitions(-DNANO_DEBUG)
endif()

# scalar type
if((NANO_WITH_FLOAT_SCALAR) AND (NOT NANO_WITH_DOUBLE_SCALAR) AND (NOT NANO_WITH_LONG_DOUBLE_SCALAR))
        message("++ Using float as the default scalar type.")
        add_definitions(-DNANO_FLOAT_SCALAR)
elseif((NOT NANO_WITH_FLOAT_SCALAR) AND (NANO_WITH_DOUBLE_SCALAR) AND (NOT NANO_WITH_LONG_DOUBLE_SCALAR))
        message("++ Using double as the default scalar type.")
        add_definitions(-DNANO_DOUBLE_SCALAR)
elseif((NOT NANO_WITH_FLOAT_SCALAR) AND (NOT NANO_WITH_DOUBLE_SCALAR) AND (NANO_WITH_LONG_DOUBLE_SCALAR))
        message("++ Using long double as the default scalar type.")
        add_definitions(-DNANO_LONG_DOUBLE_SCALAR)
else()
        message(FATAL_ERROR "++ The scalar type is not specified! Use one of the NANO_WITH_[FLOAT|DOUBLE|LONG_DOUBLE]_SCALAR options.")
endif()

# setup ctest with valgrind
setup_valgrind()
