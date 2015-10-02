# build library
file(GLOB libcortex_sources ${CMAKE_CURRENT_SOURCE_DIR}
        *.*)

add_library(cortex SHARED ${libcortex_sources})
target_link_libraries(cortex
        ${Boost_SERIALIZATION_LIBRARY}
        thread
        core)

# install library
install(TARGETS cortex
        DESTINATION lib)

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        DESTINATION include
        FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp")


                
