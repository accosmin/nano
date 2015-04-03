# no library to build!
file(GLOB liboptimize_sources
        *.*
        )

# install headers
install(DIRECTORY ${nanocv_SOURCE_DIR}/liboptimize/
        DESTINATION include/liboptimize
        FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp")


                
