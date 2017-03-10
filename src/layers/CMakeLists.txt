list(APPEND libnano_sources
        ${CMAKE_CURRENT_SOURCE_DIR}/convolution.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/affine.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/conv3d_toeplitz.cpp)

set(libnano_sources "${libnano_sources}" PARENT_SCOPE)
