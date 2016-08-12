# function to create an application
function(build_app app libs)
        get_filename_component(app_name ${app} NAME_WE)
        add_executable(${app_name} ${app})
        target_link_libraries(${app_name} ${libs})
endfunction()

# build command line programs
build_app(benchmark_conv.cpp nano)
build_app(benchmark_eigen.cpp nano)
build_app(benchmark_batch.cpp nano)
build_app(benchmark_stoch.cpp nano)
build_app(benchmark_models.cpp nano)
build_app(benchmark_iterator.cpp nano)
build_app(benchmark_trainers.cpp nano)
build_app(benchmark_functions.cpp nano)
build_app(embed.cpp nano)
build_app(info_archive.cpp nano)
build_app(info.cpp nano)
build_app(info_task.cpp nano)
build_app(train.cpp nano)
build_app(evaluate.cpp nano)
build_app(warp.cpp nano)
build_app(stats.cpp nano)
build_app(generate_convnets.cpp nano)

# install (some) programs
set(programs
        info
        info_task
        train
        evaluate
        generate_convnets)

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
