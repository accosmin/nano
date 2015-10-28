list(APPEND libcortex_sources
        ${CMAKE_CURRENT_SOURCE_DIR}/clonable.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/header.h
        ${CMAKE_CURRENT_SOURCE_DIR}/logger.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/logger.h
        ${CMAKE_CURRENT_SOURCE_DIR}/manager.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/measure_and_log.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/measure.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/row_comp.h
        ${CMAKE_CURRENT_SOURCE_DIR}/row.h
        ${CMAKE_CURRENT_SOURCE_DIR}/row_mark.h
        ${CMAKE_CURRENT_SOURCE_DIR}/singleton.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/table.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/table.h
        ${CMAKE_CURRENT_SOURCE_DIR}/timer.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/timer.h)

set(libcortex_sources "${libcortex_sources}" PARENT_SCOPE)

                
