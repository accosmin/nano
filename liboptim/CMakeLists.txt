# build library
file(GLOB liboptim_sources ${CMAKE_CURRENT_SOURCE_DIR}
        *.*)

#add_library(optim STATIC ${liboptim_sources})
#set_target_properties(optim PROPERTIES LINKER_LANGUAGE CXX)

# install library
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        DESTINATION include
        FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp")


                
