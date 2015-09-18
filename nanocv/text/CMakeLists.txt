# build library
file(GLOB libtext_sources ${CMAKE_CURRENT_SOURCE_DIR}
        *.*)

add_library(text INTERFACE)
target_include_directories(text INTERFACE
        $<BUILD_INTERFACE:${libtext_sources}>)

# install library
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        DESTINATION include
        FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp")


                
