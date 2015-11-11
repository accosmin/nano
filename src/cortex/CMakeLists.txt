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
        evaluate.cpp
        evaluate.h
        generate.cpp
        generate.h
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
        sample.cpp
        sample.h
        sampler.cpp
        sampler.h
        scalar.h
        stochastic.cpp
        stochastic.h
        string.h
        task.cpp
        task.h
        tensor.h
        trainer.cpp
        trainer_data.cpp
        trainer_data.h
        trainer.h
        trainer_result.cpp
        trainer_result.h
        trainer_state.cpp
        trainer_state.h
        version.cpp
        version.h)

add_subdirectory(criteria)
add_subdirectory(layers)
add_subdirectory(losses)
add_subdirectory(models)
add_subdirectory(tasks)
add_subdirectory(trainers)
add_subdirectory(util)
add_subdirectory(vision)

add_library(cortex SHARED ${libcortex_sources})
target_link_libraries(cortex
        ${IL_LIBRARIES}
        ${Boost_SERIALIZATION_LIBRARY}
        math
        tensor
        text
        file
        thread)

# install library
install(TARGETS cortex
        DESTINATION lib)

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        DESTINATION include
        FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp")


                
