# build functions utility library
file(GLOB libfunc_sources ${CMAKE_CURRENT_SOURCE_DIR}
        *.*)

add_library(func SHARED ${libfunc_sources})
target_link_libraries(func PRIVATE nanocv)
