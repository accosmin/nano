# build library
set(libnano_sources
        ${CMAKE_CURRENT_SOURCE_DIR}/arch.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/function.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/function_state.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/accumulator.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/logger.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/loss.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/layer.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/model.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/trainer.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/trainer_state.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/trainer_result.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/task.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/task_util.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/iterator.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/solver_batch.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/solver_stoch.cpp)

add_subdirectory(io)
add_subdirectory(math)
add_subdirectory(text)
add_subdirectory(batch)
add_subdirectory(stoch)
add_subdirectory(functions)
add_subdirectory(losses)
add_subdirectory(layers)
add_subdirectory(models)
add_subdirectory(tasks)
add_subdirectory(trainers)
add_subdirectory(iterators)
add_subdirectory(tensor)
add_subdirectory(thread)
add_subdirectory(chrono)
add_subdirectory(vision)

add_library(nano SHARED ${libnano_sources})
target_link_libraries(nano ${IL_LIBRARIES} ${ZLIB_LIBRARIES} ${BZIP2_LIBRARIES} ${LibArchive_LIBRARIES})

# install library
install(TARGETS nano
        DESTINATION lib)

install(DIRECTORY
        DESTINATION include
        FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp")
