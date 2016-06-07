list(APPEND libcortex_sources
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
        ${CMAKE_CURRENT_SOURCE_DIR}/pooling_full.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/pooling_full.h
        ${CMAKE_CURRENT_SOURCE_DIR}/pooling_soft.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/pooling_soft.h
        ${CMAKE_CURRENT_SOURCE_DIR}/pooling_gauss.h
        ${CMAKE_CURRENT_SOURCE_DIR}/pooling_gauss.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/make_layers.h)

set(libcortex_sources "${libcortex_sources}" PARENT_SCOPE)


