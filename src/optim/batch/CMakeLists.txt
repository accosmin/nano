list(APPEND libmath_sources
        ${CMAKE_CURRENT_SOURCE_DIR}/cgd.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/cgd_steps.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/gd.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/lbfgs.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/loop.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/ls_backtrack.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/ls_bisection.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/ls_cgdescent.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/ls_cubic.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/ls_init.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/ls_interpolate.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/ls_quadratic.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/ls_step.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/ls_strategy.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/params.hpp)

set(libmath_sources "${libmath_sources}" PARENT_SCOPE)
