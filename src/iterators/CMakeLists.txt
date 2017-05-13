list(APPEND libnano_sources
        ${CMAKE_CURRENT_SOURCE_DIR}/iterator_warp.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/iterator_noise.cpp)

set(libnano_sources "${libnano_sources}" PARENT_SCOPE)
