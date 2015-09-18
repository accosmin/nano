# build library
file(GLOB libtensor_sources ${CMAKE_CURRENT_SOURCE_DIR}
        *.*)

add_library(tensor INTERFACE)
target_include_directories(tensor INTERFACE
        $<BUILD_INTERFACE:${libtensor_sources}>)

# install headers
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        DESTINATION include
        FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp")


                
