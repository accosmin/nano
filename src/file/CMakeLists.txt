# build library
set(libfile_sources
        ${CMAKE_CURRENT_SOURCE_DIR}/archive.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/archive.h
        ${CMAKE_CURRENT_SOURCE_DIR}/buffer.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/buffer.h
        ${CMAKE_CURRENT_SOURCE_DIR}/bzip.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/bzip.h
        ${CMAKE_CURRENT_SOURCE_DIR}/gzip.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/gzip.h
        ${CMAKE_CURRENT_SOURCE_DIR}/mat5.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/mat5.h
        ${CMAKE_CURRENT_SOURCE_DIR}/mstream.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/mstream.h
        ${CMAKE_CURRENT_SOURCE_DIR}/ibstream.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/ibstream.h
        ${CMAKE_CURRENT_SOURCE_DIR}/obstream.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/obstream.h)

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


                
