list(APPEND libnano_sources
        ${CMAKE_CURRENT_SOURCE_DIR}/enhancer_warp.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/enhancer_noise.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/enhancer_default.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/enhancer_noclass.cpp)

set(libnano_sources "${libnano_sources}" PARENT_SCOPE)
