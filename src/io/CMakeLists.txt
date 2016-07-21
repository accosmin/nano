list(APPEND libnano_sources
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

set(libnano_sources "${libnano_sources}" PARENT_SCOPE)
