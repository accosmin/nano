# build library
file(GLOB libcore_sources ${CMAKE_CURRENT_SOURCE_DIR}
        *.*)

set(libs "${libs};${IL_LIBRARIES}")
set(libs "${libs};${ZLIB_LIBRARIES}")
set(libs "${libs};${BZIP2_LIBRARIES}")
set(libs "${libs};${LibArchive_LIBRARIES}")

add_library(core SHARED ${libcore_sources})
target_link_libraries(core ${libs})

# install library
install(TARGETS core
        DESTINATION lib)

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        DESTINATION include
        FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp")


                
