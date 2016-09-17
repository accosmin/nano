#include "utest.hpp"
#include "tensor.h"
#include "opencl/ocl.h"
#include "math/random.hpp"
#include "math/epsilon.hpp"
#include "tensor/numeric.hpp"

using namespace nano;

auto rng_size = nano::make_rng<tensor_size_t>(1, 137);
auto rng_value = nano::make_rng<scalar_t>(scalar_t(-0.1), scalar_t(+0.1));
auto n_tests = 11;

scalar_t make_scalar()
{
        return rng_value();
}

vector_t make_vector(const tensor_size_t dims)
{
        vector_t x(dims);
        tensor::set_random(rng_value, x);
        return x;
}

matrix_t make_matrix(const tensor_size_t rows, const tensor_size_t cols)
{
        matrix_t x(rows, cols);
        tensor::set_random(rng_value, x);
        return x;
}

NANO_BEGIN_MODULE(test_opencl)

// initialize OpenCL context
NANO_REQUIRE_NOTHROW(ocl::select(CL_DEVICE_TYPE_GPU));

// I/O
NANO_CASE(io)
{
        for (int test = 0; test < n_tests; ++ test)
        {
                const auto dims = rng_size();

                auto x = make_vector(dims);
                auto y = make_vector(dims);

                cl::Buffer buffer = ocl::make_buffer(x, CL_MEM_WRITE_ONLY);

                NANO_CHECK(ocl::write(buffer, x) == CL_SUCCESS);
                NANO_CHECK(ocl::read(buffer, y) == CL_SUCCESS);
                NANO_CHECK_EIGEN_CLOSE(x, y, nano::epsilon0<scalar_t>());
        }
}

// level1: vector-vector operations
NANO_CASE(level1)
{
        for (int test = 0; test < n_tests; ++ test)
        {
                const auto dims = rng_size();

                auto a = make_scalar();
                auto b = make_scalar();
                auto c = make_scalar();
                auto x = make_vector(dims);
                auto y = make_vector(dims);
                auto z = make_vector(dims);

                cl::Buffer xbuffer = ocl::make_buffer(x, CL_MEM_READ_WRITE);
                cl::Buffer ybuffer = ocl::make_buffer(y, CL_MEM_READ_WRITE);
                cl::Buffer zbuffer = ocl::make_buffer(z, CL_MEM_READ_ONLY);

                NANO_CHECK(ocl::write(xbuffer, x) == CL_SUCCESS);
                NANO_CHECK(ocl::write(ybuffer, y) == CL_SUCCESS);

                // z = x + c
                {
                        cl::Kernel kernel = ocl::make_kernel("vpc");
                        NANO_REQUIRE_NOTHROW(ocl::set_args(kernel, xbuffer, c, zbuffer, dims));
                        ocl::wait(ocl::enqueue(kernel, dims));
                        NANO_CHECK(ocl::read(zbuffer, z) == CL_SUCCESS);
                        NANO_CHECK_EIGEN_CLOSE(x.array() + c, z.array(), nano::epsilon0<scalar_t>());
                }
                // z = x + y
                {
                        cl::Kernel kernel = ocl::make_kernel("vpv");
                        NANO_REQUIRE_NOTHROW(ocl::set_args(kernel, xbuffer, ybuffer, zbuffer, dims));
                        ocl::wait(ocl::enqueue(kernel, dims));
                        NANO_CHECK(ocl::read(zbuffer, z) == CL_SUCCESS);
                        NANO_CHECK_EIGEN_CLOSE(x.array() + y.array(), z.array(), nano::epsilon0<scalar_t>());
                }
                // z = a * x + c
                {
                        cl::Kernel kernel = ocl::make_kernel("vcpc");
                        NANO_REQUIRE_NOTHROW(ocl::set_args(kernel, xbuffer, a, c, zbuffer, dims));
                        ocl::wait(ocl::enqueue(kernel, dims));
                        NANO_CHECK(ocl::read(zbuffer, z) == CL_SUCCESS);
                        NANO_CHECK_EIGEN_CLOSE(x.array() * a + c, z.array(), nano::epsilon0<scalar_t>());
                }
                // z = a * x + y
                {
                        cl::Kernel kernel = ocl::make_kernel("vcpv");
                        NANO_REQUIRE_NOTHROW(ocl::set_args(kernel, xbuffer, a, ybuffer, zbuffer, dims));
                        ocl::wait(ocl::enqueue(kernel, dims));
                        NANO_CHECK(ocl::read(zbuffer, z) == CL_SUCCESS);
                        NANO_CHECK_EIGEN_CLOSE(x.array() * a + y.array(), z.array(), nano::epsilon0<scalar_t>());
                }
                // z = a * x + b * y
                {
                        cl::Kernel kernel = ocl::make_kernel("vcpvc");
                        NANO_REQUIRE_NOTHROW(ocl::set_args(kernel, xbuffer, a, ybuffer, b, zbuffer, dims));
                        ocl::wait(ocl::enqueue(kernel, dims));
                        NANO_CHECK(ocl::read(zbuffer, z) == CL_SUCCESS);
                        NANO_CHECK_EIGEN_CLOSE(x.array() * a + y.array() * b, z.array(), nano::epsilon0<scalar_t>());
                }
                // z = a * x + b * y + c
                {
                        cl::Kernel kernel = ocl::make_kernel("vcpvcpc");
                        NANO_REQUIRE_NOTHROW(ocl::set_args(kernel, xbuffer, a, ybuffer, b, c, zbuffer, dims));
                        ocl::wait(ocl::enqueue(kernel, dims));
                        NANO_CHECK(ocl::read(zbuffer, z) == CL_SUCCESS);
                        NANO_CHECK_EIGEN_CLOSE(x.array() * a + y.array() * b + c, z.array(), nano::epsilon0<scalar_t>());
                }
        }
}

// level2: matrix-vector operations
NANO_CASE(level2)
{
        for (int test = 0; test < n_tests; ++ test)
        {
                const auto rows = rng_size();
                const auto cols = rng_size();

                auto A = make_matrix(rows, cols);
                auto x = make_vector(cols);
                auto y = make_vector(rows);
                auto z = make_vector(rows);
                auto c = make_scalar();

                cl::Buffer Abuffer = ocl::make_buffer(A, CL_MEM_READ_WRITE);
                cl::Buffer xbuffer = ocl::make_buffer(x, CL_MEM_READ_WRITE);
                cl::Buffer ybuffer = ocl::make_buffer(y, CL_MEM_READ_WRITE);
                cl::Buffer zbuffer = ocl::make_buffer(z, CL_MEM_READ_ONLY);

                NANO_CHECK(ocl::write(Abuffer, A) == CL_SUCCESS);
                NANO_CHECK(ocl::write(xbuffer, x) == CL_SUCCESS);
                NANO_CHECK(ocl::write(ybuffer, y) == CL_SUCCESS);

                // z = A * x
                {
                        cl::Kernel kernel = ocl::make_kernel("mv");
                        NANO_REQUIRE_NOTHROW(ocl::set_args(kernel, Abuffer, xbuffer, zbuffer, rows, cols));
                        ocl::wait(ocl::enqueue(kernel, rows));
                        NANO_CHECK(ocl::read(zbuffer, z) == CL_SUCCESS);
                        NANO_CHECK_EIGEN_CLOSE((A * x), z, nano::epsilon0<scalar_t>());
                }
                // z = A * x + c
                {
                        cl::Kernel kernel = ocl::make_kernel("mvpc");
                        NANO_REQUIRE_NOTHROW(ocl::set_args(kernel, Abuffer, xbuffer, c, zbuffer, rows, cols));
                        ocl::wait(ocl::enqueue(kernel, rows));
                        NANO_CHECK(ocl::read(zbuffer, z) == CL_SUCCESS);
                        NANO_CHECK_EIGEN_CLOSE((A * x).array() + c, z.array(), nano::epsilon0<scalar_t>());
                }
                // z = A * x + y
                {
                        cl::Kernel kernel = ocl::make_kernel("mvpv");
                        NANO_REQUIRE_NOTHROW(ocl::set_args(kernel, Abuffer, xbuffer, ybuffer, zbuffer, rows, cols));
                        ocl::wait(ocl::enqueue(kernel, rows));
                        NANO_CHECK(ocl::read(zbuffer, z) == CL_SUCCESS);
                        NANO_CHECK_EIGEN_CLOSE((A * x + y), z, nano::epsilon0<scalar_t>());
                }
        }
}

// level3: matrix-matrix operations
NANO_CASE(level3)
{
        for (int test = 0; test < n_tests; ++ test)
        {
                const auto rowsA = rng_size();
                const auto colsA = rng_size();
                const auto rowsB = colsA;
                const auto colsB = rng_size();

                auto A = make_matrix(rowsA, colsA);
                auto B = make_matrix(rowsB, colsB);
                auto Z = make_matrix(rowsA, colsB);

                cl::Buffer Abuffer = ocl::make_buffer(A, CL_MEM_READ_WRITE);
                cl::Buffer Bbuffer = ocl::make_buffer(B, CL_MEM_READ_WRITE);
                cl::Buffer Zbuffer = ocl::make_buffer(Z, CL_MEM_READ_ONLY);

                NANO_CHECK(ocl::write(Abuffer, A) == CL_SUCCESS);
                NANO_CHECK(ocl::write(Bbuffer, B) == CL_SUCCESS);

                // Z = A * B
                {
                        cl::Kernel kernel = ocl::make_kernel("mm");
                        NANO_REQUIRE_NOTHROW(ocl::set_args(kernel, Abuffer, Bbuffer, Zbuffer, rowsA, colsA, colsB));
                        ocl::wait(ocl::enqueue(kernel, rowsA, colsB));
                        NANO_CHECK(ocl::read(Zbuffer, Z) == CL_SUCCESS);
                        NANO_CHECK_EIGEN_CLOSE((A * B), Z, nano::epsilon0<scalar_t>());
                }
        }
}

NANO_END_MODULE()
