list(APPEND libcortex_sources
        ${CMAKE_CURRENT_SOURCE_DIR}/layer_activation.h
        ${CMAKE_CURRENT_SOURCE_DIR}/layer_activation_snorm.h
        ${CMAKE_CURRENT_SOURCE_DIR}/layer_activation_splus.h
        ${CMAKE_CURRENT_SOURCE_DIR}/layer_activation_tanh.h
        ${CMAKE_CURRENT_SOURCE_DIR}/layer_activation_unit.h
        ${CMAKE_CURRENT_SOURCE_DIR}/layer_convolution_kernel2d.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/layer_convolution_kernel2d.h
        ${CMAKE_CURRENT_SOURCE_DIR}/layer_convolution_toeplitz.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/layer_convolution_toeplitz.h
        ${CMAKE_CURRENT_SOURCE_DIR}/layer_affine.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/layer_affine.h
        ${CMAKE_CURRENT_SOURCE_DIR}/layer_pooling_full.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/layer_pooling_full.h
        ${CMAKE_CURRENT_SOURCE_DIR}/layer_pooling_soft.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/layer_pooling_soft.h
        ${CMAKE_CURRENT_SOURCE_DIR}/layer_pooling_gauss.h
        ${CMAKE_CURRENT_SOURCE_DIR}/layer_pooling_gauss.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/make_layers.h)

set(libcortex_sources "${libcortex_sources}" PARENT_SCOPE)


