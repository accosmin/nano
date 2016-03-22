if(APPLE)
        execute_process(
                COMMAND "brew info --installed qt5 | grep \"^$(brew --cellar qt5)\" | cut -d\' \' -f1"
                OUTPUT_VARIABLE QT5_DIR)
        message("${QT5_DIR}")
        set(Qt5Widgets_DIR /usr/local/Cellar/qt5/5.6.0/lib/cmake/Qt5Widgets)
endif()


find_package(Qt5Widgets REQUIRED)

set(CMAKE_AUTOMOC ON)
set(CMAKE_INCLUDE_CURRENT_DIR ON)

add_executable(info_task info_task.cpp configwin.cpp)
target_link_libraries(info_task Qt5::Widgets)

