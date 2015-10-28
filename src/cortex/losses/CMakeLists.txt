list(APPEND libcortex_sources
        ${CMAKE_CURRENT_SOURCE_DIR}/loss_cauchy.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/loss_cauchy.h
        ${CMAKE_CURRENT_SOURCE_DIR}/loss_classnll.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/loss_classnll.h
        ${CMAKE_CURRENT_SOURCE_DIR}/loss_logistic.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/loss_logistic.h
        ${CMAKE_CURRENT_SOURCE_DIR}/loss_square.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/loss_square.h)

set(libcortex_sources "${libcortex_sources}" PARENT_SCOPE)

                
