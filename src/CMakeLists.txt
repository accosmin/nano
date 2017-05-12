# build library
set(libnano_sources
        arch.cpp
        nano.cpp
        logger.cpp)

add_subdirectory(io)
add_subdirectory(math)
add_subdirectory(text)
add_subdirectory(optim)
add_subdirectory(cortex)
add_subdirectory(tensor)
add_subdirectory(thread)
add_subdirectory(vision)

add_library(nano SHARED ${libnano_sources})
target_link_libraries(nano ${IL_LIBRARIES} ${ZLIB_LIBRARIES} ${BZIP2_LIBRARIES} ${LibArchive_LIBRARIES})

# install library
install(TARGETS nano
        DESTINATION lib)

install(DIRECTORY
        DESTINATION include
        FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp")
