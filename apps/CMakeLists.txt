# build command line programs
file(GLOB nanocv_apps RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
        *.cpp)

foreach(app ${nanocv_apps})
        get_filename_component(app_name ${app} NAME_WE)

        add_executable(${app_name} ${app})
        target_link_libraries(${app_name}
                ${Boost_PROGRAM_OPTIONS_LIBRARY}
                ${Boost_FILESYSTEM_LIBRARY}
                ${Boost_SYSTEM_LIBRARY}
                nanocv)
endforeach()

# install (some) programs
set(nanocv_programs
        max_threads
        info
        info_task
        train
        evaluate
        generate
        image2cpp)

install(TARGETS
        ${nanocv_programs}
        DESTINATION bin)
                
#set_target_properties(
#        ${nanocv_programs}
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
