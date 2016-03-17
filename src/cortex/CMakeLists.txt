# build library
set(libcortex_sources
        accumulator.cpp
        accumulator.h
        batch.cpp
        batch.h
        class.cpp
        class.h
        cortex.cpp
        cortex.h
        criterion.cpp
        criterion.h
        layer.cpp
        layer.h
        loss.cpp
        loss.h
        minibatch.cpp
        minibatch.h
        model.cpp
        model.h
        optimizer.h
        protocol.h
        stochastic.cpp
        stochastic.h
        task.cpp
        task.h
        trainer.cpp
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

add_library(cortex SHARED ${libcortex_sources})
target_link_libraries(cortex vision io thread)

# install library
install(TARGETS cortex
        DESTINATION lib)

install(DIRECTORY
        DESTINATION include
        FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp")



