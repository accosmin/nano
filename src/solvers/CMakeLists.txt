list(APPEND libnano_sources
        ${CMAKE_CURRENT_SOURCE_DIR}/lsearch.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/lsearch_init.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/lsearch_length.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/solver_gd.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/solver_cgd.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/solver_nag.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/solver_lbfgs.cpp)

set(libnano_sources "${libnano_sources}" PARENT_SCOPE)
