# build library
file(GLOB libfunc_sources ${CMAKE_CURRENT_SOURCE_DIR}
        *.*)

add_library(func INTERFACE)
target_include_directories(func INTERFACE
        $<BUILD_INTERFACE:${libfunc_sources}>)

# install library
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        DESTINATION include
        FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp")