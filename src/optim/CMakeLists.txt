# build library
set(liboptim_sources
        ${CMAKE_CURRENT_SOURCE_DIR}/batch.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/problem.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/state.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/stoch.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/test.cpp)

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

