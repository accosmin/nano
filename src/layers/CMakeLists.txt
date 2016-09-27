list(APPEND libnano_sources
        ${CMAKE_CURRENT_SOURCE_DIR}/activation.h
        ${CMAKE_CURRENT_SOURCE_DIR}/activation_snorm.h
        ${CMAKE_CURRENT_SOURCE_DIR}/activation_splus.h
        ${CMAKE_CURRENT_SOURCE_DIR}/activation_tanh.h
        ${CMAKE_CURRENT_SOURCE_DIR}/activation_unit.h
        ${CMAKE_CURRENT_SOURCE_DIR}/convolution_kernel2d.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/convolution_kernel2d.h
        ${CMAKE_CURRENT_SOURCE_DIR}/convolution_toeplitz.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/convolution_toeplitz.h
        ${CMAKE_CURRENT_SOURCE_DIR}/affine.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/affine.h
        ${CMAKE_CURRENT_SOURCE_DIR}/make_layers.h)

set(libnano_sources "${libnano_sources}" PARENT_SCOPE)
