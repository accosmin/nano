list(APPEND libnano_sources
        ${CMAKE_CURRENT_SOURCE_DIR}/affine.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/normalize.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/activation.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/convolution.cpp)

set(libnano_sources "${libnano_sources}" PARENT_SCOPE)
