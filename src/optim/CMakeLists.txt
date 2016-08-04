list(APPEND libnano_sources
        ${CMAKE_CURRENT_SOURCE_DIR}/batch.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/problem.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/state.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/test.cpp)

add_subdirectory(batch)
add_subdirectory(stoch)
add_subdirectory(funcs)

set(libnano_sources "${libnano_sources}" PARENT_SCOPE)

