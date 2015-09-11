# build library
file(GLOB libtensor_sources ${CMAKE_CURRENT_SOURCE_DIR}
        *.*)

# install library
#

# install headers
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        DESTINATION include/libtensor
        FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp")


                
