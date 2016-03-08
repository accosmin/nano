# build library
set(libvision_sources
        ${CMAKE_CURRENT_SOURCE_DIR}/bilinear.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/color.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/color.h
        ${CMAKE_CURRENT_SOURCE_DIR}/convolve.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/geom.h
        ${CMAKE_CURRENT_SOURCE_DIR}/gradient.hpp
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

add_library(vision SHARED ${libvision_sources})
target_link_libraries(vision ${IL_LIBRARIES} math text tensor)

# install library
install(TARGETS vision
        DESTINATION lib)

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        DESTINATION include
        FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp")



