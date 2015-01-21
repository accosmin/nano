include_directories(
        ../src
        ../src/nanocv)

file(GLOB nanocv_apps RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *.cpp)

foreach(app ${nanocv_apps})
        get_filename_component(app_name ${app} NAME_WE)

        add_executable(${app_name} ${app})
        target_link_libraries(${app_name} nanocv)
endforeach()
