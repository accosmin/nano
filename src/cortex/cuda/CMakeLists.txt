# build library
set(libcuda_sources
        ${CMAKE_CURRENT_SOURCE_DIR}/cuda.cu
        ${CMAKE_CURRENT_SOURCE_DIR}/conv2d.cu)

cuda_add_library(libcortex_cuda STATIC ${libcuda_sources})

