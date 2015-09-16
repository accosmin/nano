# build library
file(GLOB libnanocv_sources ${CMAKE_CURRENT_SOURCE_DIR}
        *.*
        criteria/*.*
        layers/*.*
        losses/*.*
        models/*.*
        tasks/*.*
        trainers/*.*)

set(libs "${libs};${Boost_PROGRAM_OPTIONS_LIBRARY}")
set(libs "${libs};${Boost_SYSTEM_LIBRARY}")
set(libs "${libs};${Boost_FILESYSTEM_LIBRARY}")
set(libs "${libs};${Boost_IOSTREAMS_LIBRARY}")

add_library(nanocv SHARED ${libnanocv_sources})
target_link_libraries(nanocv ${libs} ml)

# install library
install(TARGETS nanocv
        DESTINATION lib)

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        DESTINATION include
        FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp")


                
