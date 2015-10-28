# build library
set(libthread_sources
        ${CMAKE_CURRENT_SOURCE_DIR}/loopi.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/loopit.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/pool.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/pool.h
        ${CMAKE_CURRENT_SOURCE_DIR}/tasks.h
        ${CMAKE_CURRENT_SOURCE_DIR}/thread.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/thread.h
        ${CMAKE_CURRENT_SOURCE_DIR}/worker.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/worker.h)

add_library(thread SHARED ${libthread_sources})

# install library
install(TARGETS thread
        DESTINATION lib)

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        DESTINATION include
        FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp")


                
