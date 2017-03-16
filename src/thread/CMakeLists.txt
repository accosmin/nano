list(APPEND libnano_sources
        ${CMAKE_CURRENT_SOURCE_DIR}/pool.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/worker.cpp)

set(libnano_sources "${libnano_sources}" PARENT_SCOPE)
