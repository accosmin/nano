# build command line programs
make_app(benchmark_eigen.cpp nano)
make_app(benchmark_batch.cpp nano)
make_app(benchmark_stoch.cpp nano)
make_app(benchmark_tpool.cpp nano)
make_app(benchmark_conv3d.cpp nano)
make_app(benchmark_models.cpp nano)
make_app(benchmark_enhancer.cpp nano)
make_app(benchmark_functions.cpp nano)

make_app(info.cpp nano)
make_app(info_mat5.cpp nano)
make_app(info_task.cpp nano)
make_app(info_model.cpp nano)
make_app(info_archive.cpp nano)
make_app(info_enhancer.cpp nano)

make_app(train.cpp nano)
make_app(evaluate.cpp nano)

make_app(embed.cpp nano)
make_app(stats.cpp "")
make_app(tabulate.cpp nano)

# install (some) programs
set(programs
        info
        info_mat5
        info_task
        info_model
        info_archive
        info_enhancer
        train
        evaluate
        embed
        stats
        tabulate)

install(TARGETS ${programs} DESTINATION bin)

#set_target_properties(
#        ${programs}
#        PROPERTIES INSTALL_RPATH "../lib:./:./lib")

## http://www.cmake.org/Wiki/CMake_RPATH_handling
#set(CMAKE_SKIP_BUILD_RPATH  FALSE)
#set(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)
#set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")
#set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

#list(FIND CMAKE_PLATFORM_IMPLICIT_LINK_DIRECTORIES "${CMAKE_INSTALL_PREFIX}/lib" isSystemDir)
#if("${isSystemDir}" STREQUAL "-1")
#        set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")
#endif("${isSystemDir}" STREQUAL "-1")
