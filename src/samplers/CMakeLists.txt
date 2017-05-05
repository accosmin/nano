list(APPEND libnano_sources
        ${CMAKE_CURRENT_SOURCE_DIR}/sampler_none.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/sampler_warp.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/sampler_noise.cpp)

set(libnano_sources "${libnano_sources}" PARENT_SCOPE)
