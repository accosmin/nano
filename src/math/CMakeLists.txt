list(APPEND libnano_sources
        ${CMAKE_CURRENT_SOURCE_DIR}/average.h
        ${CMAKE_CURRENT_SOURCE_DIR}/cubic.h
        ${CMAKE_CURRENT_SOURCE_DIR}/epsilon.h
        ${CMAKE_CURRENT_SOURCE_DIR}/gauss.h
        ${CMAKE_CURRENT_SOURCE_DIR}/momentum.h
        ${CMAKE_CURRENT_SOURCE_DIR}/numeric.h
        ${CMAKE_CURRENT_SOURCE_DIR}/quadratic.h
        ${CMAKE_CURRENT_SOURCE_DIR}/random.h
        ${CMAKE_CURRENT_SOURCE_DIR}/stats.h
        ${CMAKE_CURRENT_SOURCE_DIR}/softmax.h
        ${CMAKE_CURRENT_SOURCE_DIR}/tune.h
        ${CMAKE_CURRENT_SOURCE_DIR}/tune_result.h
        ${CMAKE_CURRENT_SOURCE_DIR}/tune_grid_space.h
        ${CMAKE_CURRENT_SOURCE_DIR}/tune_finite_space.h)

set(libnano_sources "${libnano_sources}" PARENT_SCOPE)
