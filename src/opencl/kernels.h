#pragma once

namespace nano
{
        inline const char* opencl_kernels()
        {
                return R"xxx(

                // add two vectors: z = x + y
                __kernel void add_vv(
                        __global const float* x,
                        __global const float* y,
                        __global float* z)
                {
                        const int i = get_global_id(0);
                        z[i] = x[i] + y[i];
                }

                // multiply a matrix by a vector: y = A * x
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

                // multiply a matrix by a matrix: C = A * B
                __kernel void mul_mm(
                        __global const float* A, const int colsA,
                        __global const float* B, const int colsB,
                        __global float* C)
                {
                        const int rowA = get_global_id(0);
                        const int colB = get_global_id(1);
                        float sum = 0;
                        for (int colA = 0; colA < colsA; ++ colA)
                        {
                                sum += A[rowA * colsA + colA] * B[colA * colsB + colB];
                        }
                        C[rowA * colsB + colB] = sum;
                }

                )xxx";
        }
}

