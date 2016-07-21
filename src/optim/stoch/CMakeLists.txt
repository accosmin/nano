list(APPEND libnano_sources
        ${CMAKE_CURRENT_SOURCE_DIR}/adadelta.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/adagrad.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/adam.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/ag.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/sg.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/sgm.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/ngd.cpp)

set(libnano_sources "${libnano_sources}" PARENT_SCOPE)

