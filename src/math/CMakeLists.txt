# build library
set(libmath_sources
        ${CMAKE_CURRENT_SOURCE_DIR}/abs.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/cast.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/clamp.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/close.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/epsilon.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/gauss.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/numeric.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/random.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/stats.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/usampling.hpp)

add_library(math INTERFACE)
target_include_directories(math INTERFACE
        $<BUILD_INTERFACE:${libmath_sources}>)

# install library
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        DESTINATION include
        FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp")


                
