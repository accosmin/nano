# build test utility library
file(GLOB nanocv-test_sources
        *.*)

add_library(nanocv-test SHARED ${nanocv-test_sources})
target_link_libraries(nanocv-test nanocv)
