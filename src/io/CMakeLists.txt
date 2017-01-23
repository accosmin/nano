list(APPEND libnano_sources
        ${CMAKE_CURRENT_SOURCE_DIR}/io.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/archive.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/istream.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/istream_mem.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/istream_zlib.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/istream_archive.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/mat5.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/ibstream.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/obstream.cpp)

set(libnano_sources "${libnano_sources}" PARENT_SCOPE)
