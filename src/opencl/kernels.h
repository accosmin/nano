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

                float vsum4(const float4 in)
                {
                        const float4 unit = 1.0f;
                        return dot(in, unit);
                }

                float dotx(__global const float* x, __global const float* y, const int size)
                {
                        float acc = 0.0f;
                        for (int i = 0; i < size; ++ i)
                        {
                                acc += x[i] * y[i];
                        }
                        return acc;
                }

                float dotx4(__global const float* x, __global const float* y, const int size)
                {
                        float4 acc = 0.0f;

                        const int tail = size & 3;
                        const int size4 = size - tail;
                        for (int i = 0; i < size4; i += 4)
                        {
                                acc += vload4(0, &x[i]) * vload4(0, &y[i]);
                        }

                        return (!tail) ? vsum4(acc) : vsum4(acc) + dotx(x + size4, y + size4, tail);
                }

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
                        z[row] = dotx4(&A[row * cols], x, cols);
                }

                // multiply a matrix by a vector and add a constant: y = A * x + c
                __kernel void mvpc(
                        __global const float* A, const int cols,
                        __global const float* x,
                        const float c,
                        __global float* z)
                {
                        const int row = get_global_id(0);
                        z[row] = dotx4(&A[row * cols], x, cols) + c;
                }

                // multiply a matrix by a vector and add a vector: y = A * x + y
                __kernel void mvpv(
                        __global const float* A, const int cols,
                        __global const float* x,
                        __global const float* y,
                        __global float* z)
                {
                        const int row = get_global_id(0);
                        z[row] = dotx4(&A[row * cols], x, cols) + y[row];
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

