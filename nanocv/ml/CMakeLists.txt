# build library
file(GLOB libml_sources ${CMAKE_CURRENT_SOURCE_DIR}
        *.*)

add_library(ml SHARED ${libml_sources})
target_link_libraries(ml core)

# install library
install(TARGETS ml
        DESTINATION lib)

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        DESTINATION include
        FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp")


                
