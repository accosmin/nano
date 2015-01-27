add_subdirectory(libnanocv-test)

# setup CMake to run tests
enable_testing()

# build unit tests
file(GLOB test_apps RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *.cpp)

foreach(test_app ${test_apps})
        get_filename_component(test_name ${test_app} NAME_WE)

        add_executable(${test_name} ${test_app})
        target_link_libraries(${test_name} ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY} libnanocv libnanocv-test)

        add_test(${test_name} ${test_name})
endforeach()
