# build library
file(GLOB libmath_sources ${CMAKE_CURRENT_SOURCE_DIR}
        *.*)

add_library(math INTERFACE)
target_include_directories(math INTERFACE
        $<BUILD_INTERFACE:${libmath_sources}>)

# install library
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        DESTINATION include
        FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp")


                
