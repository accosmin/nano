# build library
file(GLOB libthread_sources ${CMAKE_CURRENT_SOURCE_DIR}
        *.*)

add_library(thread SHARED ${libthread_sources})

# install library
install(TARGETS thread
        DESTINATION lib)

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        DESTINATION include
        FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp")


                
