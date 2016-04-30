list(APPEND libcortex_sources
        ${CMAKE_CURRENT_SOURCE_DIR}/layer_activation.h
        ${CMAKE_CURRENT_SOURCE_DIR}/layer_activation_snorm.h
        ${CMAKE_CURRENT_SOURCE_DIR}/layer_activation_splus.h
        ${CMAKE_CURRENT_SOURCE_DIR}/layer_activation_tanh.h
        ${CMAKE_CURRENT_SOURCE_DIR}/layer_activation_unit.h
        ${CMAKE_CURRENT_SOURCE_DIR}/layer_convolution.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/layer_convolution.h
        ${CMAKE_CURRENT_SOURCE_DIR}/layer_affine.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/layer_affine.h
        ${CMAKE_CURRENT_SOURCE_DIR}/layer_pool.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/layer_pool.h
        ${CMAKE_CURRENT_SOURCE_DIR}/layer_pooling_ada3x3.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/layer_pooling_ada3x3.h
        ${CMAKE_CURRENT_SOURCE_DIR}/layer_pooling_gauss.h
        ${CMAKE_CURRENT_SOURCE_DIR}/layer_pooling_gauss.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/layer_pooling_adaexp.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/layer_pooling_adaexp.h
        ${CMAKE_CURRENT_SOURCE_DIR}/pooling.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/make_layers.h)

set(libcortex_sources "${libcortex_sources}" PARENT_SCOPE)


