list(APPEND libnano_sources
        ${CMAKE_CURRENT_SOURCE_DIR}/algorithm.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/cmdline.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/table.cpp)

set(libnano_sources "${libnano_sources}" PARENT_SCOPE)
