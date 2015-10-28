list(APPEND libcortex_sources
        ${CMAKE_CURRENT_SOURCE_DIR}/avg_criterion.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/avg_criterion.h
        ${CMAKE_CURRENT_SOURCE_DIR}/avg_l2_criterion.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/avg_l2_criterion.h
        ${CMAKE_CURRENT_SOURCE_DIR}/avg_var_criterion.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/avg_var_criterion.h)

set(libcortex_sources "${libcortex_sources}" PARENT_SCOPE)

                
