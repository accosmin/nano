#include "cuda.h"

namespace ncv
{
        bool cuda::copyToDevice(const double* h_data, int size, double* d_data)
        {
                return false;
        }

        bool cuda::copyFromDevice(const double* d_data, int size, double* h_data)
        {
                return false;
        }
}
