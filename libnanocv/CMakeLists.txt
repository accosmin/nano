# build library
file(GLOB libnanocv_sources ${CMAKE_CURRENT_SOURCE_DIR}
        *.*
        criteria/*.*
        file/*.*
        functions/*.*
        layers/*.*
        losses/*.*
        math/*.*
        models/*.*
        tasks/*.*
        text/*.*
        thread/*.*
        trainers/*.*
        vision/*.*)

set(libs "${libs};${Boost_SERIALIZATION_LIBRARY}")
set(libs "${libs};${Boost_PROGRAM_OPTIONS_LIBRARY}")
set(libs "${libs};${Boost_SYSTEM_LIBRARY}")
set(libs "${libs};${Boost_FILESYSTEM_LIBRARY}")
set(libs "${libs};${Boost_IOSTREAMS_LIBRARY}")
set(libs "${libs};${ZLIB_LIBRARIES}")
set(libs "${libs};${BZIP2_LIBRARIES}")
set(libs "${libs};${LibArchive_LIBRARIES}")
set(libs "${libs};${IL_LIBRARIES}")

add_library(nanocv SHARED ${libnanocv_sources})
target_link_libraries(nanocv ${libs})

# install library
install(TARGETS
        nanocv
        DESTINATION lib)

# install headers
install(DIRECTORY ${nanocv_SOURCE_DIR}/nanocv/
        DESTINATION include/nanocv
        FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp")


                
