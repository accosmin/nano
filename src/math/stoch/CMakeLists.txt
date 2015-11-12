list(APPEND libmath_sources
        ${CMAKE_CURRENT_SOURCE_DIR}/adadelta.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/adagrad.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/ag.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/ag_restarts.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/best_state.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/decay.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/params.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/sga.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/sg.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/sia.hpp)

set(libmath_sources "${libmath_sources}" PARENT_SCOPE)


                
