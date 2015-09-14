# build library
file(GLOB libtensor_sources ${CMAKE_CURRENT_SOURCE_DIR}
        *.*)

add_library(tensor STATIC ${libtensor_sources})
set_target_properties(tensor PROPERTIES LINKER_LANGUAGE CXX)

# install headers
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        DESTINATION include
        FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp")


                
