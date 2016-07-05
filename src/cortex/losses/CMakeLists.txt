list(APPEND libcortex_sources
        ${CMAKE_CURRENT_SOURCE_DIR}/cauchy.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/cauchy.h
        ${CMAKE_CURRENT_SOURCE_DIR}/classnll.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/classnll.h
        ${CMAKE_CURRENT_SOURCE_DIR}/exponential.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/exponential.h
        ${CMAKE_CURRENT_SOURCE_DIR}/logistic.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/logistic.h
        ${CMAKE_CURRENT_SOURCE_DIR}/square.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/square.h)

set(libcortex_sources "${libcortex_sources}" PARENT_SCOPE)

                
