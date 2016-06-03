list(APPEND liboptim_sources
        ${CMAKE_CURRENT_SOURCE_DIR}/adadelta.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/adagrad.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/adam.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/ag.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/loop.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/lrate.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/params.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/sg.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/sgm.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/ngd.hpp)

set(liboptim_sources "${liboptim_sources}" PARENT_SCOPE)

