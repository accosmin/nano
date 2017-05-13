list(APPEND libnano_sources
        ${CMAKE_CURRENT_SOURCE_DIR}/cgd.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/gd.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/lbfgs.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/ls_backtrack.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/ls_cgdescent.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/ls_init.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/ls_step.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/ls_interpolate.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/ls_strategy.cpp)

set(libnano_sources "${libnano_sources}" PARENT_SCOPE)
