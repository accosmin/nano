list(APPEND libnano_sources
        ${CMAKE_CURRENT_SOURCE_DIR}/archive.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/buffer.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/mem_reader.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/zlib_reader.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/archive_reader.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/mat5.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/ibstream.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/obstream.cpp)

set(libnano_sources "${libnano_sources}" PARENT_SCOPE)
