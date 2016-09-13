#pragma once

namespace nano
{
        ///
        /// \brief basic linear algebra kernels:
        ///     c - constant
        //      v - vector
        ///     m - matrix
        ///     p - plus (addition)
        ///
        inline const char* opencl_kernels()
        {
                return R"xxx(

                // add a constant to a vector: z = x + c
                __kernel void vpc(
                        __global const float* x,
                        const float c,
                        __global float* z)
                {
                        const int i = get_global_id(0);
                        z[i] = x[i] + c;
                }

                // add two vectors: z = x + y
                __kernel void vpv(
                        __global const float* x,
                        __global const float* y,
                        __global float* z)
                {
                        const int i = get_global_id(0);
                        z[i] = x[i] + y[i];
                }

                // add two vectors with multiplication factors: z = a * x + b * y
                __kernel void vcpvc(
                        __global const float* x, const float a,
                        __global const float* y, const float b,
                        __global float* z)
                {
                        const int i = get_global_id(0);
                        z[i] = a * x[i] + b * y[i];
                }

                // multiply a matrix by a vector: z = A * x
                __kernel void mv(
                        __global const float* A, const int cols,
                        __global const float* x,
                        __global float* z)
                {
                        const int row = get_global_id(0);
                        float sum = 0;
                        for (int col = 0; col < cols; ++ col)
                        {
                                sum += A[row * cols + col] * x[col];
                        }
                        z[row] = sum;
                }

                // multiply a matrix by a vector and add a constant: y = A * x + c
                __kernel void mvpc(
                        __global const float* A, const int cols,
                        __global const float* x,
                        const float c,
                        __global float* z)
                {
                        const int row = get_global_id(0);
                        float sum = 0;
                        for (int col = 0; col < cols; ++ col)
                        {
                                sum += A[row * cols + col] * x[col];
                        }
                        z[row] = sum + c;
                }

                // multiply a matrix by a vector and add a vector: y = A * x + y
                __kernel void mvpv(
                        __global const float* A, const int cols,
                        __global const float* x,
                        __global const float* y,
                        __global float* z)
                {
                        const int row = get_global_id(0);
                        float sum = 0;
                        for (int col = 0; col < cols; ++ col)
                        {
                                sum += A[row * cols + col] * x[col];
                        }
                        z[row] = sum + y[row];
                }

                // multiply a matrix by a matrix: Z = A * B
                __kernel void mm(
                        __global const float* A, const int colsA,
                        __global const float* B, const int colsB,
                        __global float* Z)
                {
                        const int rowA = get_global_id(0);
                        const int colB = get_global_id(1);
                        float sum = 0;
                        for (int colA = 0; colA < colsA; ++ colA)
                        {
                                sum += A[rowA * colsA + colA] * B[colA * colsB + colB];
                        }
                        Z[rowA * colsB + colB] = sum;
                }

                )xxx";
        }
}

