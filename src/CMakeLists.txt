# build library
set(libnano_sources
        ${CMAKE_CURRENT_SOURCE_DIR}/arch.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/loss.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/task.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/model.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/solver.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/function.cpp)

add_subdirectory(core)
add_subdirectory(tasks)
add_subdirectory(losses)
add_subdirectory(models)
add_subdirectory(tensor)
add_subdirectory(solvers)
add_subdirectory(functions)

# NB: let the use toggle the library type with BUILD_SHARED_LIBS=ON/OFF
add_library(nano ${libnano_sources})
target_link_libraries(nano ${IL_LIBRARIES} ${ZLIB_LIBRARIES} ${BZIP2_LIBRARIES} ${LibArchive_LIBRARIES})

# install library
install(TARGETS nano
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
        RUNTIME DESTINATION bin
        DESTINATION lib)

install(DIRECTORY
        DESTINATION include
        FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp")
