# build library
file(GLOB libnanocv_sources ${CMAKE_CURRENT_SOURCE_DIR}
        *.*
        criteria/*.*
        layers/*.*
        losses/*.*
        models/*.*
        tasks/*.*
        trainers/*.*)

add_library(nanocv SHARED ${libnanocv_sources})
target_link_libraries(nanocv
        cortex)

# install library
install(TARGETS nanocv
        DESTINATION lib)

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        DESTINATION include
        FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp")


                
