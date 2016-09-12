#pragma once

namespace nano
{
        inline const char* opencl_kernels()
        {
                return R"xxx(

                // adds two vectors: z = x + y
                __kernel void add_vv(
                        __global const float* x,
                        __global const float* y,
                        __global float* z)
                {
                        const int i = get_global_id(0);
                        z[i] = x[i] + y[i];
                }

                // multiple a matrix by a vector: y = A * x
                __kernel void mul_mv(
                        __global const float* A, const int cols,
                        __global const float* x,
                        __global float* y)
                {
                        const int row = get_global_id(0);
                        float sum = 0;
                        for (int col = 0; col < cols; ++ col)
                        {
                                sum += A[row * cols + col] * x[col];
                        }
                        y[row] = sum;
                }

                )xxx";
        }
}

