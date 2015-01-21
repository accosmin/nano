# Build library
file(GLOB nanocv_sources
        *.*
        file/*.*
        util/*.*
        tensor/*.*
        optimize/*.*
        nanocv/*.*
        nanocv/criteria/*.*
        nanocv/losses/*.*
        nanocv/layers/*.*
        nanocv/tasks/*.*
        nanocv/models/*.*
        nanocv/trainers/*.*
        )

if(NANOCV_WITH_OPENCL)
        file(GLOB nanocv_opencl_sources
                opencl/*.*
        )
        set(nanocv_sources "${nanocv_sources};${nanocv_opencl_sources}")
endif()

set(libs "${libs};${Boost_SERIALIZATION_LIBRARY}")
set(libs "${libs};${Boost_PROGRAM_OPTIONS_LIBRARY}")
set(libs "${libs};${Boost_SYSTEM_LIBRARY}")
set(libs "${libs};${Boost_FILESYSTEM_LIBRARY}")
set(libs "${libs};${Boost_IOSTREAMS_LIBRARY}")
set(libs "${libs};${ZLIB_LIBRARIES}")
set(libs "${libs};${BZIP2_LIBRARIES}")
set(libs "${libs};${LibArchive_LIBRARIES}")
set(libs "${libs};${IL_LIBRARIES}")
if(NANOCV_WITH_OPENCL)
        set(libs "${libs};${OpenCL_LIBRARIES}")
endif()

add_library(nanocv
        SHARED
        ${nanocv_sources})
target_link_libraries(nanocv ${libs})

if(NANOCV_WITH_CUDA)
        file(GLOB nanocv_cuda_sources
                cuda/*.*
        )

        cuda_add_library(nanocv_cuda STATIC ${nanocv_cuda_sources})
        target_link_libraries(nanocv nanocv_cuda)
endif()

# Install library
install(TARGETS
        nanocv
        DESTINATION lib)

# Install headers
install(DIRECTORY ${nanocv_SOURCE_DIR}/src/
        DESTINATION include/nanocv
        FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp")


                
