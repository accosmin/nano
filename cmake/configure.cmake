include(cmake/utils.cmake)

# setup compiler (gcc or clang)
if(CMAKE_CXX_COMPILER_ID MATCHES GNU OR CMAKE_CXX_COMPILER_ID MATCHES Clang)
        message("++ Compiling with ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION} ...")

        require_cpp14()

        # set flags
        set(CMAKE_CXX_FLAGS                     "-std=c++14 -pedantic -march=native -mtune=native -ffunction-sections")
        set(CMAKE_CXX_FLAGS                     "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wconversion")
        set(CMAKE_CXX_FLAGS                     "${CMAKE_CXX_FLAGS} -Woverloaded-virtual")
if(CMAKE_CXX_COMPILER_ID MATCHES GNU)
        set(CMAKE_CXX_FLAGS                     "${CMAKE_CXX_FLAGS} -Wmisleading-indentation")
else()
        set(CMAKE_CXX_FLAGS                     "${CMAKE_CXX_FLAGS} -Wno-missing-braces")
endif()
        set(CMAKE_CXX_FLAGS                     "${CMAKE_CXX_FLAGS} -Wreorder")
        set(CMAKE_CXX_FLAGS                     "${CMAKE_CXX_FLAGS} -Wnon-virtual-dtor")
        set(CMAKE_CXX_FLAGS                     "${CMAKE_CXX_FLAGS} -Wcast-align")
        set(CMAKE_CXX_FLAGS                     "${CMAKE_CXX_FLAGS} -Wold-style-cast")
        set(CMAKE_CXX_FLAGS                     "${CMAKE_CXX_FLAGS} -Wshadow")
        set(CMAKE_CXX_FLAGS                     "${CMAKE_CXX_FLAGS} -Wsign-promo")
        set(CMAKE_CXX_FLAGS                     "${CMAKE_CXX_FLAGS} -Wsign-conversion")
        set(CMAKE_CXX_FLAGS                     "${CMAKE_CXX_FLAGS} -fno-common")
        set(CMAKE_CXX_FLAGS                     "${CMAKE_CXX_FLAGS} -fvisibility=hidden")
        set(CMAKE_CXX_FLAGS_DEBUG               "-g -fno-omit-frame-pointer")
        set(CMAKE_CXX_FLAGS_RELEASE             "-O3 -DNDEBUG -DEIGEN_NO_DEBUG -DEIGEN_DONT_PARALLELIZE")
        set(CMAKE_CXX_FLAGS_RELWITHDEBINFO      "-O3 -g -fno-omit-frame-pointer")
        set(CMAKE_CXX_FLAGS_MINSIZEREL          "-Os -DNDEBUG")

        if(NOT CMAKE_CXX_COMPILER_ID MATCHES AppleClang)
                set(CMAKE_CXX_FLAGS             "${CMAKE_CXX_FLAGS} -pthread")
        endif()

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
