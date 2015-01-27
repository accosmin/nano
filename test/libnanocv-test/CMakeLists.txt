# build test utility library
file(GLOB libnanocv-test_sources
        *.*)

add_library(libnanocv-test SHARED ${libnanocv-test_sources})
target_link_libraries(libnanocv-test libnanocv)
