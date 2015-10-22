# build library
file(GLOB libfile_sources ${CMAKE_CURRENT_SOURCE_DIR}
        *.*)

add_library(file SHARED ${libfile_sources})
target_link_libraries(file
        ${ZLIB_LIBRARIES}
        ${BZIP2_LIBRARIES}
        ${LibArchive_LIBRARIES})

# install library
install(TARGETS file
        DESTINATION lib)

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        DESTINATION include
        FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp")


                
