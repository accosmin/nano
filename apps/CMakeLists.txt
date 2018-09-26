# build command line utilities
make_app(info.cpp nano)
make_app(info_mat5.cpp nano)
make_app(info_archive.cpp nano)

make_app(train.cpp nano)
make_app(evaluate.cpp nano)

make_app(stats.cpp "")
make_app(tabulate.cpp nano)

make_app(bench_eigen.cpp nano)
make_app(bench_tpool.cpp nano)
make_app(bench_model.cpp nano)
make_app(bench_conv3d.cpp nano)
make_app(bench_affine.cpp nano)
make_app(bench_solvers.cpp nano)
make_app(bench_functions.cpp nano)

# install (some) programs
set(programs
        info
        info_mat5
        info_archive
        train
        evaluate
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
