#include "utest.hpp"
#include "tensor.h"
#include "math/random.hpp"
#include "opencl/kernels.h"
#include "opencl/manager.h"
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
opencl_manager_t theocl;
NANO_REQUIRE_NOTHROW(theocl.init());
NANO_REQUIRE_NOTHROW(theocl.select(CL_DEVICE_TYPE_GPU));

// create supported kernels
cl::Program program;
NANO_REQUIRE_NOTHROW(program = theocl.make_program_from_text(opencl_kernels()));

cl::Kernel kernel_vpc;
cl::Kernel kernel_vpv;
cl::Kernel kernel_vcpvc;

cl::Kernel kernel_mv;
cl::Kernel kernel_mvpc;
cl::Kernel kernel_mvpv;

cl::Kernel kernel_mm;

NANO_REQUIRE_NOTHROW(kernel_vpc = theocl.make_kernel(program, "vpc"));
NANO_REQUIRE_NOTHROW(kernel_vpv = theocl.make_kernel(program, "vpv"));
NANO_REQUIRE_NOTHROW(kernel_vcpvc = theocl.make_kernel(program, "vcpvc"));

NANO_REQUIRE_NOTHROW(kernel_mv = theocl.make_kernel(program, "mv"));
NANO_REQUIRE_NOTHROW(kernel_mvpc = theocl.make_kernel(program, "mvpc"));
NANO_REQUIRE_NOTHROW(kernel_mvpv = theocl.make_kernel(program, "mvpv"));

NANO_REQUIRE_NOTHROW(kernel_mm = theocl.make_kernel(program, "mm"));

// use this command queue to send tasks
cl::CommandQueue& queue = theocl.command_queue();

// z = x + c
NANO_CASE(vpc)
{
        for (int test = 0; test < n_tests; ++ test)
        {
                const auto dims = rng_size();
                auto c = make_scalar();
                auto x = make_vector(dims);
                auto z = make_vector(dims);

                cl::Buffer xbuffer = theocl.make_buffer(x, CL_MEM_READ_WRITE);
                cl::Buffer zbuffer = theocl.make_buffer(z, CL_MEM_READ_ONLY);

                NANO_REQUIRE_NOTHROW(nano::set_args(kernel_vpc, xbuffer, c, zbuffer));

                NANO_CHECK(theocl.write(xbuffer, x) == CL_SUCCESS);

                queue.enqueueNDRangeKernel(kernel_vpc, cl::NullRange, cl::NDRange(size_t(dims)), cl::NullRange);
                queue.finish();

                NANO_CHECK(theocl.read(zbuffer, z) == CL_SUCCESS);
                NANO_CHECK_EIGEN_CLOSE(x.array() + c, z.array(), nano::epsilon0<scalar_t>());
        }
}

// z = x + y
NANO_CASE(vpv)
{
        for (int test = 0; test < n_tests; ++ test)
        {
                const auto dims = rng_size();
                auto x = make_vector(dims);
                auto y = make_vector(dims);
                auto z = make_vector(dims);

                cl::Buffer xbuffer = theocl.make_buffer(x, CL_MEM_READ_WRITE);
                cl::Buffer ybuffer = theocl.make_buffer(y, CL_MEM_READ_WRITE);
                cl::Buffer zbuffer = theocl.make_buffer(z, CL_MEM_READ_ONLY);

                NANO_REQUIRE_NOTHROW(nano::set_args(kernel_vpv, xbuffer, ybuffer, zbuffer));

                NANO_CHECK(theocl.write(xbuffer, x) == CL_SUCCESS);
                NANO_CHECK(theocl.write(ybuffer, y) == CL_SUCCESS);

                queue.enqueueNDRangeKernel(kernel_vpv, cl::NullRange, cl::NDRange(size_t(dims)), cl::NullRange);
                queue.finish();

                NANO_CHECK(theocl.read(zbuffer, z) == CL_SUCCESS);
                NANO_CHECK_EIGEN_CLOSE((x + y), z, nano::epsilon0<scalar_t>());
        }
}

// z = a * x + b * y
NANO_CASE(vcpvc)
{
        for (int test = 0; test < n_tests; ++ test)
        {
                const auto dims = rng_size();
                auto a = make_scalar();
                auto b = make_scalar();
                auto x = make_vector(dims);
                auto y = make_vector(dims);
                auto z = make_vector(dims);

                cl::Buffer xbuffer = theocl.make_buffer(x, CL_MEM_READ_WRITE);
                cl::Buffer ybuffer = theocl.make_buffer(y, CL_MEM_READ_WRITE);
                cl::Buffer zbuffer = theocl.make_buffer(z, CL_MEM_READ_ONLY);

                NANO_REQUIRE_NOTHROW(nano::set_args(kernel_vcpvc, xbuffer, a, ybuffer, b, zbuffer));

                NANO_CHECK(theocl.write(xbuffer, x) == CL_SUCCESS);
                NANO_CHECK(theocl.write(ybuffer, y) == CL_SUCCESS);

                queue.enqueueNDRangeKernel(kernel_vcpvc, cl::NullRange, cl::NDRange(size_t(dims)), cl::NullRange);
                queue.finish();

                NANO_CHECK(theocl.read(zbuffer, z) == CL_SUCCESS);
                NANO_CHECK_EIGEN_CLOSE((a * x + b * y), z, nano::epsilon0<scalar_t>());
        }
}

// Z = A * x
NANO_CASE(mv)
{
        for (int test = 0; test < n_tests; ++ test)
        {
                const auto rows = rng_size();
                const auto cols = rng_size();
                auto A = make_matrix(rows, cols);
                auto x = make_vector(cols);
                auto z = make_vector(rows);

                cl::Buffer Abuffer = theocl.make_buffer(A, CL_MEM_READ_WRITE);
                cl::Buffer xbuffer = theocl.make_buffer(x, CL_MEM_READ_WRITE);
                cl::Buffer zbuffer = theocl.make_buffer(z, CL_MEM_READ_ONLY);

                NANO_REQUIRE_NOTHROW(nano::set_args(kernel_mv, Abuffer, static_cast<int>(cols), xbuffer, zbuffer));

                NANO_CHECK(theocl.write(Abuffer, A) == CL_SUCCESS);
                NANO_CHECK(theocl.write(xbuffer, x) == CL_SUCCESS);

                queue.enqueueNDRangeKernel(kernel_mv, cl::NullRange, cl::NDRange(size_t(rows)), cl::NullRange);
                queue.finish();

                NANO_CHECK(theocl.read(zbuffer, z) == CL_SUCCESS);
                NANO_CHECK_EIGEN_CLOSE((A * x), z, nano::epsilon0<scalar_t>());
        }
}

// Z = A * x + c
NANO_CASE(mvpc)
{
        for (int test = 0; test < n_tests; ++ test)
        {
                const auto rows = rng_size();
                const auto cols = rng_size();
                auto c = make_scalar();
                auto A = make_matrix(rows, cols);
                auto x = make_vector(cols);
                auto z = make_vector(rows);

                cl::Buffer Abuffer = theocl.make_buffer(A, CL_MEM_READ_WRITE);
                cl::Buffer xbuffer = theocl.make_buffer(x, CL_MEM_READ_WRITE);
                cl::Buffer zbuffer = theocl.make_buffer(z, CL_MEM_READ_ONLY);

                NANO_REQUIRE_NOTHROW(nano::set_args(kernel_mvpc, Abuffer, static_cast<int>(cols), xbuffer, c, zbuffer));

                NANO_CHECK(theocl.write(Abuffer, A) == CL_SUCCESS);
                NANO_CHECK(theocl.write(xbuffer, x) == CL_SUCCESS);

                queue.enqueueNDRangeKernel(kernel_mvpc, cl::NullRange, cl::NDRange(size_t(rows)), cl::NullRange);
                queue.finish();

                NANO_CHECK(theocl.read(zbuffer, z) == CL_SUCCESS);
                NANO_CHECK_EIGEN_CLOSE((A * x).array() + c, z.array(), nano::epsilon0<scalar_t>());
        }
}

// Z = A * x + y
NANO_CASE(mvpv)
{
        for (int test = 0; test < n_tests; ++ test)
        {
                const auto rows = rng_size();
                const auto cols = rng_size();
                auto A = make_matrix(rows, cols);
                auto x = make_vector(cols);
                auto y = make_vector(rows);
                auto z = make_vector(rows);

                cl::Buffer Abuffer = theocl.make_buffer(A, CL_MEM_READ_WRITE);
                cl::Buffer xbuffer = theocl.make_buffer(x, CL_MEM_READ_WRITE);
                cl::Buffer ybuffer = theocl.make_buffer(y, CL_MEM_READ_WRITE);
                cl::Buffer zbuffer = theocl.make_buffer(z, CL_MEM_READ_ONLY);

                NANO_REQUIRE_NOTHROW(nano::set_args(kernel_mvpv, Abuffer, static_cast<int>(cols), xbuffer, ybuffer, zbuffer));

                NANO_CHECK(theocl.write(Abuffer, A) == CL_SUCCESS);
                NANO_CHECK(theocl.write(xbuffer, x) == CL_SUCCESS);
                NANO_CHECK(theocl.write(ybuffer, y) == CL_SUCCESS);

                queue.enqueueNDRangeKernel(kernel_mvpv, cl::NullRange, cl::NDRange(size_t(rows)), cl::NullRange);
                queue.finish();

                NANO_CHECK(theocl.read(zbuffer, z) == CL_SUCCESS);
                NANO_CHECK_EIGEN_CLOSE((A * x + y), z, nano::epsilon0<scalar_t>());
        }
}

// Z = A * B
NANO_CASE(mm)
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

                cl::Buffer Abuffer = theocl.make_buffer(A, CL_MEM_READ_WRITE);
                cl::Buffer Bbuffer = theocl.make_buffer(B, CL_MEM_READ_WRITE);
                cl::Buffer Zbuffer = theocl.make_buffer(Z, CL_MEM_READ_ONLY);

                NANO_REQUIRE_NOTHROW(nano::set_args(kernel_mm, Abuffer, static_cast<int>(colsA), Bbuffer, static_cast<int>(colsB), Zbuffer));

                NANO_CHECK(theocl.write(Abuffer, A) == CL_SUCCESS);
                NANO_CHECK(theocl.write(Bbuffer, B) == CL_SUCCESS);

                queue.enqueueNDRangeKernel(kernel_mm, cl::NullRange, cl::NDRange(size_t(rowsA), size_t(colsB)), cl::NullRange);
                queue.finish();

                NANO_CHECK(theocl.read(Zbuffer, Z) == CL_SUCCESS);
                NANO_CHECK_EIGEN_CLOSE((A * B), Z, nano::epsilon0<scalar_t>());
        }
}

NANO_END_MODULE()
