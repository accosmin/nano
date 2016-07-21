# build library
set(libcortex_sources
        accumulator.cpp
        accumulator.h
        class.cpp
        class.h
        cortex.cpp
        cortex.h
        criterion.cpp
        criterion.h
        layer.h
        loss.h
        model.cpp
        model.h
        protocol.h
        task.h
        task_util.h
        task_util.cpp
        task_iterator.h
        trainer.h
        trainer_result.cpp
        trainer_result.h
        trainer_state.cpp
        trainer_state.h
        version.cpp
        version.h
        clonable.hpp
        logger.cpp
        logger.h
        manager.hpp
        measure_and_log.hpp
        measure.hpp
        timer.cpp
        timer.h)

add_subdirectory(criteria)
add_subdirectory(layers)
add_subdirectory(losses)
add_subdirectory(models)
add_subdirectory(tasks)
add_subdirectory(trainers)
if(NANO_WITH_OPENCL)
        add_subdirectory(opencl)
endif()

add_library(cortex SHARED ${libcortex_sources})
target_link_libraries(cortex vision io thread optim)
if(NANO_WITH_OPENCL)
        target_link_libraries(cortex ${OpenCL_LIBRARIES})
endif()

# install library
install(TARGETS cortex
        DESTINATION lib)

install(DIRECTORY
        DESTINATION include
        FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp")



