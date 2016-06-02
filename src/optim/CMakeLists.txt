# build library
set(liboptim_sources
        ${CMAKE_CURRENT_SOURCE_DIR}/batch.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/batch_types.h
        ${CMAKE_CURRENT_SOURCE_DIR}/ls_types.h
        ${CMAKE_CURRENT_SOURCE_DIR}/problem.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/problem.h
        ${CMAKE_CURRENT_SOURCE_DIR}/state.h
        ${CMAKE_CURRENT_SOURCE_DIR}/stoch.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/stoch_types.h)

add_subdirectory(batch)
add_subdirectory(stoch)
add_subdirectory(funcs)

add_library(optim SHARED ${liboptim_sources})
target_link_libraries(optim math text tensor)

# install library
install(TARGETS optim
        DESTINATION lib)

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        DESTINATION include
        FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp")

