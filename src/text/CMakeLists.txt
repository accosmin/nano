list(APPEND libnano_sources
        ${CMAKE_CURRENT_SOURCE_DIR}/align.h
        ${CMAKE_CURRENT_SOURCE_DIR}/algorithm.h
        ${CMAKE_CURRENT_SOURCE_DIR}/algorithm.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/filesystem.h
        ${CMAKE_CURRENT_SOURCE_DIR}/filesystem.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/cmdline.h
        ${CMAKE_CURRENT_SOURCE_DIR}/cmdline.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/concatenate.h
        ${CMAKE_CURRENT_SOURCE_DIR}/enum_string.h
        ${CMAKE_CURRENT_SOURCE_DIR}/from_params.h
        ${CMAKE_CURRENT_SOURCE_DIR}/from_string.h
        ${CMAKE_CURRENT_SOURCE_DIR}/to_params.h
        ${CMAKE_CURRENT_SOURCE_DIR}/to_string.h
        ${CMAKE_CURRENT_SOURCE_DIR}/table.h
        ${CMAKE_CURRENT_SOURCE_DIR}/table.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/table_header.h
        ${CMAKE_CURRENT_SOURCE_DIR}/table_row.h)

set(libnano_sources "${libnano_sources}" PARENT_SCOPE)
