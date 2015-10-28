# build library
set(libmin_sources
        ${CMAKE_CURRENT_SOURCE_DIR}/batch.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/batch_types.h
        ${CMAKE_CURRENT_SOURCE_DIR}/lsearch_types.h
        ${CMAKE_CURRENT_SOURCE_DIR}/params.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/problem.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/state.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/stoch.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/stoch_types.h
        ${CMAKE_CURRENT_SOURCE_DIR}/tune_fixed.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/tune_log10.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/tune_log10_mt.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/tune_stoch.hpp)

add_subdirectory(batch)
add_subdirectory(stoch)
add_subdirectory(test)

add_library(min INTERFACE)
target_include_directories(min INTERFACE
        $<BUILD_INTERFACE:${libmin_sources}>)

# install library
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        DESTINATION include
        FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp")


                
