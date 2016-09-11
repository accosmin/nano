#pragma once

namespace nano
{
        namespace ocl
        {
                ///
                /// \brief kernel for adding two vectors: z = x + y
                ///
                inline const char* get_kernel_addv()
                {
                        return R"xxx(
                        __kernel void kernel_addv(
                                __global const double* x,
                                __global const double* y,
                                __global double* z)
                        {
                                const int i = get_global_id(0);
                                z[i] = x[i] + y[i];
                        })xxx";
                }
        }
}

