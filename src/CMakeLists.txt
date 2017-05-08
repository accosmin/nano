# build library
set(libnano_sources
        arch.cpp
        accumulator.cpp
        class.cpp
        nano.cpp
        "function.cpp"
        state.cpp
        task_util.cpp
        task_iterator.cpp
        trainer_state.cpp
        trainer_result.cpp
        logger.cpp)

add_subdirectory(io)
add_subdirectory(math)
add_subdirectory(text)
add_subdirectory(batch)
add_subdirectory(stoch)
add_subdirectory(tensor)
add_subdirectory(thread)
add_subdirectory(vision)
add_subdirectory(layers)
add_subdirectory(losses)
add_subdirectory(models)
add_subdirectory(tasks)
add_subdirectory(samplers)
add_subdirectory(trainers)
add_subdirectory(functions)

add_library(nano SHARED ${libnano_sources})
target_link_libraries(nano ${IL_LIBRARIES} ${ZLIB_LIBRARIES} ${BZIP2_LIBRARIES} ${LibArchive_LIBRARIES})

# install library
install(TARGETS nano
        DESTINATION lib)

install(DIRECTORY
        DESTINATION include
        FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp")



