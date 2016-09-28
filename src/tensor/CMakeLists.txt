list(APPEND libnano_sources
        ${CMAKE_CURRENT_SOURCE_DIR}/numeric.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/serialize.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/tensor.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/tensor_index.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/tensor_storage.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/algorithm.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/vector.hpp)

set(libnano_sources "${libnano_sources}" PARENT_SCOPE)

