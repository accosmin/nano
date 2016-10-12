list(APPEND libnano_sources
        ${CMAKE_CURRENT_SOURCE_DIR}/bilinear.h
        ${CMAKE_CURRENT_SOURCE_DIR}/color.h
        ${CMAKE_CURRENT_SOURCE_DIR}/convolve.h
        ${CMAKE_CURRENT_SOURCE_DIR}/geom.h
        ${CMAKE_CURRENT_SOURCE_DIR}/gradient.h
        ${CMAKE_CURRENT_SOURCE_DIR}/image.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/image_grid.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/image_grid.h
        ${CMAKE_CURRENT_SOURCE_DIR}/image.h
        ${CMAKE_CURRENT_SOURCE_DIR}/image_io.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/image_io.h
        ${CMAKE_CURRENT_SOURCE_DIR}/point.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/point.h
        ${CMAKE_CURRENT_SOURCE_DIR}/rect.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/rect.h
        ${CMAKE_CURRENT_SOURCE_DIR}/warp.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/warp.h)

set(libnano_sources "${libnano_sources}" PARENT_SCOPE)
