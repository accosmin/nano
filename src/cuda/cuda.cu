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

        namespace cuda
        {
                struct device_buffer_impl_t
                {
                        thrust::device_vector<double>   m_data;
                };
        }

        cuda::device_buffer_t::device_buffer_t(int size)
                :       m_impl(new cuda::device_buffer_impl_t)
        {
                m_impl->m_data.resize(size);
        }

        cuda::device_buffer_t::~device_buffer_t()
        {
                delete m_impl;
        }

        int cuda::device_buffer_t::size() const
        {
                return static_cast<int>(m_impl->m_data.size());
        }

        bool cuda::device_buffer_t::empty() const
        {
                return m_impl->m_data.empty();
        }

        const cuda::device_buffer_impl_t& cuda::device_buffer_t::get() const
        {
                return *m_impl;
        }

        cuda::device_buffer_impl_t& cuda::device_buffer_t::get()
        {
                return *m_impl;
        }

        bool cuda::device_buffer_t::copyToDevice(const double* h_data) const
        {
                thrust::copy(h_data, h_data + size(), m_impl->m_data.begin());
                return true;
        }

        bool cuda::device_buffer_t::copyFromDevice(double* h_data) const
        {
                thrust::copy(m_impl->m_data.begin(), m_impl->m_data.end(), h_data);
                return false;
        }

        bool cuda::addbsquared(const device_buffer_t& a, const device_buffer_t& b, device_buffer_t& c)
        {
                if (    a.size() != c.size() ||
                        b.size() != c.size())
                {
                        return false;
                }

                else
                {
                        const thrust::device_vector<double>& d_a = a.get().m_data;
                        const thrust::device_vector<double>& d_b = b.get().m_data;
                        thrust::device_vector<double>& d_c = c.get().m_data;

                        const dim3 ksize = make_size(a.size());
                        const dim3 bsize = make_block_size(a.size());

                        kernel_addbsquared<<<ksize, bsize>>>(
                                thrust::raw_pointer_cast(&d_a[0]),
                                thrust::raw_pointer_cast(&d_b[0]),
                                d_a.size(),
                                thrust::raw_pointer_cast(&d_c[0]));

                        return true;
                }
        }
}
