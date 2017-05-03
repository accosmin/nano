list(APPEND libnano_sources
        ${CMAKE_CURRENT_SOURCE_DIR}/image.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/image_grid.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/image_io.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/point.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/rect.cpp)

set(libnano_sources "${libnano_sources}" PARENT_SCOPE)
