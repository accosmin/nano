list(APPEND libnano_sources
        ${CMAKE_CURRENT_SOURCE_DIR}/align.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/algorithm.h
        ${CMAKE_CURRENT_SOURCE_DIR}/algorithm.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/filesystem.h
        ${CMAKE_CURRENT_SOURCE_DIR}/filesystem.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/cmdline.h
        ${CMAKE_CURRENT_SOURCE_DIR}/cmdline.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/concatenate.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/enum_string.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/from_params.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/from_string.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/to_params.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/to_string.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/table.h
        ${CMAKE_CURRENT_SOURCE_DIR}/table.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/table_header.h
        ${CMAKE_CURRENT_SOURCE_DIR}/table_row.h)

set(libnano_sources "${libnano_sources}" PARENT_SCOPE)
