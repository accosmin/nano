# build library
set(libtensor_sources
        ${CMAKE_CURRENT_SOURCE_DIR}/conv2d_cpp.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/conv2d_dot.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/conv2d_doti.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/conv2d_dyn.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/conv2d_eig.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/conv2d_mad.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/conv2d_madi.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/corr2d_cpp.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/corr2d_dyn.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/corr2d_egb.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/corr2d_egr.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/corr2d_mdk.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/corr2d_mdki.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/corr2d_mdo.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/corr2d_mdoi.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/for_each.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/matrix.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/random.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/serialize.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/tensor_base.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/tensor.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/tensor_map.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/transform.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/vector.hpp)

add_library(tensor INTERFACE)

add_library(tensor_dummy STATIC
        ${libtensor_sources}
        ${CMAKE_CURRENT_SOURCE_DIR}/_dummy.h
        ${CMAKE_CURRENT_SOURCE_DIR}/_dummy.cpp)

# install headers
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        DESTINATION include
        FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp")


                
