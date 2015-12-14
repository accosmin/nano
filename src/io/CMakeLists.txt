# build library
set(libio_sources
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
        ${CMAKE_CURRENT_SOURCE_DIR}/imstream.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/imstream.h
        ${CMAKE_CURRENT_SOURCE_DIR}/ibstream.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/ibstream.h
        ${CMAKE_CURRENT_SOURCE_DIR}/obstream.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/obstream.h)

add_library(io SHARED ${libio_sources})
target_link_libraries(io
        ${ZLIB_LIBRARIES}
        ${BZIP2_LIBRARIES}
        ${LibArchive_LIBRARIES}
        text)

# install library
install(TARGETS io
        DESTINATION lib)

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        DESTINATION include
        FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp")


                
