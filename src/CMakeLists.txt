# build library
set(libnano_sources
        arch.cpp
        accumulator.cpp
        class.cpp
        nano.cpp
        criterion.cpp
        model.cpp
        task_util.cpp
        trainer_result.cpp
        trainer_state.cpp
        version.cpp
        logger.cpp
        timer.cpp)

add_subdirectory(io)
add_subdirectory(math)
add_subdirectory(text)
add_subdirectory(optim)
add_subdirectory(tensor)
add_subdirectory(thread)
add_subdirectory(vision)
add_subdirectory(criteria)
add_subdirectory(layers)
add_subdirectory(losses)
add_subdirectory(models)
add_subdirectory(tasks)
add_subdirectory(trainers)
if(NANO_WITH_OPENCL)
        add_subdirectory(opencl)
endif()
if(NANO_WITH_CUDA)
        add_subdirectory(cuda)
endif()

add_library(nano SHARED ${libnano_sources})
target_link_libraries(nano ${IL_LIBRARIES} ${ZLIB_LIBRARIES} ${BZIP2_LIBRARIES} ${LibArchive_LIBRARIES})
if(NANO_WITH_OPENCL)
        target_link_libraries(nano ${OpenCL_LIBRARIES})
endif()
if(NANO_WITH_CUDA)
        target_link_libraries(nano libnano_cuda)
endif()

# install library
install(TARGETS nano
        DESTINATION lib)

install(DIRECTORY
        DESTINATION include
        FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp")



