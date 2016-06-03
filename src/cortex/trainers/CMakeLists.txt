list(APPEND libcortex_sources
        ${CMAKE_CURRENT_SOURCE_DIR}/batch_trainer.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/batch_trainer.h
        ${CMAKE_CURRENT_SOURCE_DIR}/minibatch_trainer.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/minibatch_trainer.h
        ${CMAKE_CURRENT_SOURCE_DIR}/stochastic_trainer.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/stochastic_trainer.h)

set(libcortex_sources "${libcortex_sources}" PARENT_SCOPE)


