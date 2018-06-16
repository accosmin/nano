list(APPEND libnano_sources
        ${CMAKE_CURRENT_SOURCE_DIR}/lsearch.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/solver_gd.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/solver_cgd.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/solver_nag.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/solver_lbfgs.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/ls_backtrack.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/ls_cgdescent.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/ls_interpolate.cpp)

set(libnano_sources "${libnano_sources}" PARENT_SCOPE)
