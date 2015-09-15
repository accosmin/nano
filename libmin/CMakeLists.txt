# build library
file(GLOB libmin_sources ${CMAKE_CURRENT_SOURCE_DIR}
        *.*)

#add_library(min STATIC ${libmin_sources})
#set_target_properties(min PROPERTIES LINKER_LANGUAGE CXX)

# install library
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        DESTINATION include
        FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp")


                
