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

__global__ void kernel_addbsquared(const double* a, const double* b, int size, double* c)
{
        const int i = threadIdx.x + blockIdx.x * blockDim.x;
        if (i < size)
        {
                c[i] = a[i] + b[i] * b[i];
        }
}

namespace ncv
{
        int cuda::count_devices()
        {
                int count = 0;
                CUDA_HANDLE_ERROR(cudaGetDeviceCount(&count));

                return count;
        }

        cudaDeviceProp cuda::get_device_properties(int device)
        {
                cudaDeviceProp prop;
                CUDA_HANDLE_ERROR(cudaGetDeviceProperties(&prop, device));

                return prop;
        }

        bool cuda::print_info()
        {
                const int count = cuda::count_devices();
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

        dim3 make_size(int size, int device = 0)
        {
                const cudaDeviceProp prop = cuda::get_device_properties(device);
                return dim3((size + prop.maxThreadsPerBlock - 1) / prop.maxThreadsPerBlock, 1, 1);
        }

        dim3 make_block_size(int size, int device = 0)
        {
                const cudaDeviceProp prop = cuda::get_device_properties(device);
                return dim3(prop.maxThreadsPerBlock, 1, 1);
        }

        bool cuda::addbsquared(const vector_t<double>& a, const vector_t<double>& b, vector_t<double>& c)
        {
                if (    a.size() != c.size() ||
                        b.size() != c.size())
                {
                        return false;
                }

                else
                {
                        const dim3 ksize = make_size(a.size());
                        const dim3 bsize = make_block_size(a.size());

                        kernel_addbsquared<<<ksize, bsize>>>(a.data(), b.data(), a.size(), c.data());

                        return cudaGetLastError() == cudaSuccess;
                }
        }
}
