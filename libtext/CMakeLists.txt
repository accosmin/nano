# build library
file(GLOB libtext_sources ${CMAKE_CURRENT_SOURCE_DIR}
        *.*)

#add_library(text STATIC ${libtext_sources})
#set_target_properties(text PROPERTIES LINKER_LANGUAGE CXX)

# install library
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        DESTINATION include
        FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp")


                
