#include "cuda.h"
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/copy.h>

namespace ncv
{
        bool cuda::copyToDevice(const double* h_data, int size)
        {
                thrust::device_vector<double> d_data(size);
                thrust::copy(h_data, h_data + size, d_data.begin());

                return true;
        }

        bool cuda::copyFromDevice(const double* d_data, int size, double* h_data)
        {
                return false;
        }
}
