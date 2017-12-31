list(APPEND libnano_sources
        ${CMAKE_CURRENT_SOURCE_DIR}/layer_affine.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/layer_norm3d.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/layer_conv3d.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/layer_plus4d.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/layer_tcat4d.cpp)

set(libnano_sources "${libnano_sources}" PARENT_SCOPE)
