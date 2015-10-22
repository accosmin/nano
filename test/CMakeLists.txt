# setup CMake to run tests
enable_testing()

# function to create a unit test application
function(build_test_app test_app libs)
        get_filename_component(test_name ${test_app} NAME_WE)

        add_executable(${test_name} ${test_app})
        target_link_libraries(${test_name}
                ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY}
                ${libs})

        add_test(${test_name} ${test_name})
endfunction()

# function to create a unit test application
function(build_test_apps test_apps_dir libs)
        file(GLOB test_apps RELATIVE ${test_apps_dir} *.cpp)

        foreach(test_app ${test_apps})
                build_test_app("${test_app}" "${libs}")
        endforeach()
endfunction()

# build unit tests
add_subdirectory(core)
add_subdirectory(file)
add_subdirectory(thread)
add_subdirectory(cortex)


