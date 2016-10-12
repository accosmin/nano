list(APPEND libnano_sources
        ${CMAKE_CURRENT_SOURCE_DIR}/loopi.h
        ${CMAKE_CURRENT_SOURCE_DIR}/loopit.h
        ${CMAKE_CURRENT_SOURCE_DIR}/pool.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/pool.h
        ${CMAKE_CURRENT_SOURCE_DIR}/queue.h
        ${CMAKE_CURRENT_SOURCE_DIR}/section.h
        ${CMAKE_CURRENT_SOURCE_DIR}/worker.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/worker.h)

set(libnano_sources "${libnano_sources}" PARENT_SCOPE)
