# build library
file(GLOB libcortex_sources ${CMAKE_CURRENT_SOURCE_DIR}
        *.*
        file/*.*
        table/*.*
        vision/*.*
        criteria/*.*
        layers/*.*
        losses/*.*
        models/*.*
        tasks/*.*
        trainers/*.*)

add_library(cortex SHARED ${libcortex_sources})
target_link_libraries(cortex
        ${IL_LIBRARIES}
        ${LibArchive_LIBRARIES}
        ${Boost_SERIALIZATION_LIBRARY}
        thread)

# install library
install(TARGETS cortex
        DESTINATION lib)

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        DESTINATION include
        FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp")


                
