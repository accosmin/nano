# build library
file(GLOB libnanocv_sources
        *.*
        file/*.*
        util/*.*
        tensor/*.*
        optimize/*.*
        criteria/*.*
        losses/*.*
        layers/*.*
        tasks/*.*
        models/*.*
        trainers/*.*
        )

if(NANOCV_WITH_OPENCL)
        file(GLOB libnanocv_opencl_sources
                opencl/*.*
        )
        set(libnanocv_sources "${libnanocv_sources};${libnanocv_opencl_sources}")
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

add_library(libnanocv SHARED ${libnanocv_sources})
target_link_libraries(libnanocv ${libs})

if(NANOCV_WITH_CUDA)
        file(GLOB libnanocv_cuda_sources
                cuda/*.*
        )

        cuda_add_library(libnanocv_cuda STATIC ${libnanocv_cuda_sources})
        target_link_libraries(libnanocv libnanocv_cuda)
endif()

# install library
install(TARGETS
        libnanocv
        DESTINATION lib)

# install headers
install(DIRECTORY ${nanocv_SOURCE_DIR}/src/
        DESTINATION include/nanocv
        FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp")


                
