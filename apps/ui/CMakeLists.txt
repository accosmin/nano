set(CMAKE_AUTOMOC ON)
set(CMAKE_INCLUDE_CURRENT_DIR ON)

find_package(Qt4 4.8 REQUIRED QtCore QtGui)

add_executable(info_task info_task.cpp configwin.cpp)
target_link_libraries(info_task Qt4::Gui)

