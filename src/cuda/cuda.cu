#include "cuda.h"
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/copy.h>
#include <cstdio>

static void HandleError(cudaError_t err, const char*file, int line)
{
        if (err != cudaSuccess)
        {
                printf("%s in %s at line %d\n", cudaGetErrorString(err), file, line);
                exit(EXIT_FAILURE);
        }
}

#define CUDA_HANDLE_ERROR(err) (HandleError(err, __FILE__, __LINE__))

namespace ncv
{
        const cuda::manager_t& cuda::manager_t::instance()
        {
                static const cuda::manager_t the_instance;
                return the_instance;
        }
        
        cuda::manager_t::manager_t()
                :       m_devices(0)
        {
                CUDA_HANDLE_ERROR(cudaGetDeviceCount(&m_devices));                
                
                for (int device = 0; device < m_devices; device ++)
                {
                        CUDA_HANDLE_ERROR(cudaGetDeviceProperties(&m_properties[device], device));
                }
        }   
        
        bool cuda::manager_t::print_info() const
        {
                const int count = count_devices();
                for (int i = 0; i < count; i ++)
                {
                        const cudaDeviceProp prop = get_device_properties(i);
                        
                        printf("CUDA device [%d/%d]: name = %s\n", i + 1, count, prop.name);
                        printf("CUDA device [%d/%d]: compute capability = %d.%d\n", i + 1, count, prop.major, prop.minor);
                        printf("CUDA device [%d/%d]: clock rate = %d\n", i + 1, count, prop.clockRate);
                        printf("CUDA device [%d/%d]: global mem = %ld\n", i + 1, count, prop.totalGlobalMem);
                        printf("CUDA device [%d/%d]: constant Mem = %ld\n", i + 1, count, prop.totalConstMem);
                        printf("CUDA device [%d/%d]: mem pitch = %ld\n", i + 1, count, prop.memPitch);
                        printf("CUDA device [%d/%d]: texture alignment = %ld\n", i + 1, count, prop.textureAlignment);
                        printf("CUDA device [%d/%d]: multiprocessor count = %d\n", i + 1, count, prop.multiProcessorCount);
                        printf("CUDA device [%d/%d]: shared mem per mp = %ld\n", i + 1, count, prop.sharedMemPerBlock);
                        printf("CUDA device [%d/%d]: registers per mp = %d\n", i + 1, count, prop.regsPerBlock);
                        printf("CUDA device [%d/%d]: threads in warp = %d\n", i + 1, count, prop.warpSize);
                        printf("CUDA device [%d/%d]: max threads per block = %d\n", i + 1, count, prop.maxThreadsPerBlock);
                        printf("CUDA device [%d/%d]: max thread dimensions = (%d, %d, %d)\n", i + 1, count,
                               prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
                        printf("CUDA device [%d/%d]: max grid dimensions = (%d, %d, %d)\n", i + 1, count,
                               prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
                        printf("\n");
                }
                
                return true;
        }
        
        int cuda::count_devices()
        {
                return manager_t::instance().count_devices();
        }

        cudaDeviceProp cuda::get_device_properties(int device)
        {
                return manager_t::instance().get_device_properties(device);
        }

        bool cuda::print_info()
        {
                return manager_t::instance().print_info();
        }

        dim3 cuda::make_blocks1d(int size, int device)
        {
                const cudaDeviceProp prop = cuda::get_device_properties(device);
                return dim3((size + prop.maxThreadsPerBlock - 1) / prop.maxThreadsPerBlock,
                            1,
                            1);
        }

        dim3 cuda::make_blocks2d(int rows, int cols, int device)
        {
                const cudaDeviceProp prop = cuda::get_device_properties(device);
                return dim3((cols + prop.maxThreadsPerBlock - 1) / prop.maxThreadsPerBlock,
                            (rows + prop.maxThreadsPerBlock - 1) / prop.maxThreadsPerBlock,
                            1);
        }

        dim3 cuda::make_threads1d(int, int device)
        {
                const cudaDeviceProp prop = cuda::get_device_properties(device);
                return dim3(prop.maxThreadsPerBlock,
                            1,
                            1);
        }

        dim3 cuda::make_threads2d(int, int, int device)
        {
                const cudaDeviceProp prop = cuda::get_device_properties(device);
                return dim3(sqrt(prop.maxThreadsPerBlock),
                            sqrt(prop.maxThreadsPerBlock),
                            1);
        }
}
