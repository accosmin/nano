list(APPEND libnano_sources
        ${CMAKE_CURRENT_SOURCE_DIR}/numeric.h
        ${CMAKE_CURRENT_SOURCE_DIR}/serialize.h
        ${CMAKE_CURRENT_SOURCE_DIR}/tensor.h
        ${CMAKE_CURRENT_SOURCE_DIR}/index.h
        ${CMAKE_CURRENT_SOURCE_DIR}/storage.h
        ${CMAKE_CURRENT_SOURCE_DIR}/algorithm.h
        ${CMAKE_CURRENT_SOURCE_DIR}/vector.h)

set(libnano_sources "${libnano_sources}" PARENT_SCOPE)

