list(APPEND libnano_sources
        ${CMAKE_CURRENT_SOURCE_DIR}/layer_affine.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/layer_activation.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/layer_convolution.cpp)

set(libnano_sources "${libnano_sources}" PARENT_SCOPE)
