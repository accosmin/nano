list(APPEND libnano_sources
        ${CMAKE_CURRENT_SOURCE_DIR}/conv2d_assert.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/conv2d_cpp.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/conv2d_dot.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/conv2d_dyn.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/conv2d_eig.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/conv2d_mad.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/corr2d_cpp.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/corr2d_dyn.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/corr2d_egb.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/corr2d_egr.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/corr2d_mdk.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/corr2d_mdo.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/numeric.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/serialize.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/tensor.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/tensor_index.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/tensor_storage.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/algorithm.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/vector.hpp)

set(libnano_sources "${libnano_sources}" PARENT_SCOPE)

