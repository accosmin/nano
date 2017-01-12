list(APPEND libnano_sources
        ${CMAKE_CURRENT_SOURCE_DIR}/archive.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/buffer.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/bzip.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/gzip.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/mat5.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/istream.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/imstream.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/ibstream.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/obstream.cpp)

set(libnano_sources "${libnano_sources}" PARENT_SCOPE)
