# build library
set(libmath_sources
        ${CMAKE_CURRENT_SOURCE_DIR}/abs.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/average_scalar.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/average_vector.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/cast.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/clamp.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/close.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/cubic.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/epsilon.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/gauss.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/numeric.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/quadratic.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/random.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/stats.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/usampling.hpp

        ${CMAKE_CURRENT_SOURCE_DIR}/batch.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/batch_types.h
        ${CMAKE_CURRENT_SOURCE_DIR}/lsearch_types.h
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
add_subdirectory(funcs)

add_library(math INTERFACE)

# install library
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        DESTINATION include
        FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp")


                
