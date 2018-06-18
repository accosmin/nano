list(APPEND libnano_sources
        ${CMAKE_CURRENT_SOURCE_DIR}/lsearch.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/lsearch_init_unit.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/lsearch_init_quadratic.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/lsearch_init_consistent.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/lsearch_backtrack.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/lsearch_cgdescent.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/lsearch_interpolate.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/solver_gd.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/solver_cgd.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/solver_nag.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/solver_lbfgs.cpp)

set(libnano_sources "${libnano_sources}" PARENT_SCOPE)
