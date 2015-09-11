# build library
file(GLOB libmath_sources ${CMAKE_CURRENT_SOURCE_DIR}
        *.*)

# install library
#

# install headers
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        DESTINATION include
        FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp")


                
