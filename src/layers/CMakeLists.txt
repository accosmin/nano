list(APPEND libnano_sources
        ${CMAKE_CURRENT_SOURCE_DIR}/convolution.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/affine.cpp)

set(libnano_sources "${libnano_sources}" PARENT_SCOPE)
