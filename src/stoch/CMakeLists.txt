list(APPEND libnano_sources
        ${CMAKE_CURRENT_SOURCE_DIR}/solver_stoch_adadelta.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/solver_stoch_adagrad.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/solver_stoch_adam.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/solver_stoch_adaratio.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/solver_stoch_ag.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/solver_stoch_sg.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/solver_stoch_sgm.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/solver_stoch_ngd.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/solver_stoch_svrg.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/solver_stoch_asgd.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/solver_stoch_rmsprop.cpp)

set(libnano_sources "${libnano_sources}" PARENT_SCOPE)
