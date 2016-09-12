#pragma once

namespace nano
{
        ///
        /// \brief kernel for adding two vectors: z = x + y
        ///
        inline const char* opencl_kernel_addv()
        {
                return R"xxx(
                __kernel void addv(
                        __global const float* x,
                        __global const float* y,
                        __global float* z)
                {
                        const int i = get_global_id(0);
                        z[i] = x[i] + y[i];
                })xxx";
        }
}

