# build library
file(GLOB libmin_sources ${CMAKE_CURRENT_SOURCE_DIR}
        *.*
        batch/*.*
        stoch/*.*)

add_library(min INTERFACE)
target_include_directories(min INTERFACE
        $<BUILD_INTERFACE:${libmin_sources}>)

# install library
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        DESTINATION include
        FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp")


                
