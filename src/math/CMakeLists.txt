list(APPEND libnano_sources
        ${CMAKE_CURRENT_SOURCE_DIR}/abs.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/average.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/clamp.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/close.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/cubic.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/epsilon.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/gauss.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/momentum.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/numeric.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/quadratic.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/random.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/stats.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/softmax.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/tune.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/tune_result.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/tune_grid_space.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/tune_finite_space.hpp)

set(libnano_sources "${libnano_sources}" PARENT_SCOPE)
