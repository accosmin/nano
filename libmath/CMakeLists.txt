# build library
file(GLOB libmath_sources ${CMAKE_CURRENT_SOURCE_DIR}
        *.*)

#add_library(math STATIC ${libmath_sources})
#set_target_properties(math PROPERTIES LINKER_LANGUAGE CXX)

# install library
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        DESTINATION include
        FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp")


                
