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

NANO_BEGIN_MODULE(test_opencl)

NANO_CASE(add_vv)
{
        opencl_manager_t theocl;
        NANO_REQUIRE_NOTHROW(theocl.init());
        NANO_REQUIRE_NOTHROW(theocl.select(CL_DEVICE_TYPE_GPU));

        cl::Program program;
        NANO_REQUIRE_NOTHROW(program = theocl.make_program_from_text(opencl_kernels()));

        cl::Kernel kernel;
        NANO_REQUIRE_NOTHROW(kernel = theocl.make_kernel(program, "add_vv"));

        for (int test = 0; test < n_tests; ++ test)
        {
                const auto dims = rng_size();

                vector_t x(dims), y(dims), z(dims);
                tensor::set_random(rng_value, x, y, z);

                cl::Buffer xbuffer = theocl.make_buffer(x, CL_MEM_READ_WRITE);
                cl::Buffer ybuffer = theocl.make_buffer(y, CL_MEM_READ_WRITE);
                cl::Buffer zbuffer = theocl.make_buffer(z, CL_MEM_READ_ONLY);

                kernel.setArg(0, xbuffer);
                kernel.setArg(1, ybuffer);
                kernel.setArg(2, zbuffer);

                NANO_CHECK(theocl.write(xbuffer, x) == CL_SUCCESS);
                NANO_CHECK(theocl.write(ybuffer, y) == CL_SUCCESS);

                cl::CommandQueue& queue = theocl.command_queue();
                queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(size_t(dims)), cl::NullRange);
                queue.finish();

                NANO_CHECK(theocl.read(zbuffer, z) == CL_SUCCESS);
                NANO_CHECK_EIGEN_CLOSE((x + y), z, nano::epsilon0<scalar_t>());
        }
}

NANO_CASE(mul_mv)
{
        opencl_manager_t theocl;
        NANO_REQUIRE_NOTHROW(theocl.init());
        NANO_REQUIRE_NOTHROW(theocl.select(CL_DEVICE_TYPE_GPU));

        cl::Program program;
        NANO_REQUIRE_NOTHROW(program = theocl.make_program_from_text(opencl_kernels()));

        cl::Kernel kernel;
        NANO_REQUIRE_NOTHROW(kernel = theocl.make_kernel(program, "mul_mv"));

        for (int test = 0; test < n_tests; ++ test)
        {
                const auto rows = rng_size();
                const auto cols = rng_size();

                matrix_t A(rows, cols);
                vector_t x(cols), y(rows);
                tensor::set_random(rng_value, x, y, A);

                cl::Buffer Abuffer = theocl.make_buffer(A, CL_MEM_READ_WRITE);
                cl::Buffer xbuffer = theocl.make_buffer(x, CL_MEM_READ_WRITE);
                cl::Buffer ybuffer = theocl.make_buffer(y, CL_MEM_READ_ONLY);

                kernel.setArg(0, Abuffer);
                kernel.setArg(1, static_cast<int>(cols));
                kernel.setArg(2, xbuffer);
                kernel.setArg(3, ybuffer);

                NANO_CHECK(theocl.write(Abuffer, A) == CL_SUCCESS);
                NANO_CHECK(theocl.write(xbuffer, x) == CL_SUCCESS);

                cl::CommandQueue& queue = theocl.command_queue();
                queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(size_t(rows)), cl::NullRange);
                queue.finish();

                NANO_CHECK(theocl.read(ybuffer, y) == CL_SUCCESS);
                NANO_CHECK_EIGEN_CLOSE((A * x), y, nano::epsilon0<scalar_t>());
        }
}

NANO_CASE(mul_mm)
{
        opencl_manager_t theocl;
        NANO_REQUIRE_NOTHROW(theocl.init());
        NANO_REQUIRE_NOTHROW(theocl.select(CL_DEVICE_TYPE_GPU));

        cl::Program program;
        NANO_REQUIRE_NOTHROW(program = theocl.make_program_from_text(opencl_kernels()));

        cl::Kernel kernel;
        NANO_REQUIRE_NOTHROW(kernel = theocl.make_kernel(program, "mul_mm"));

        for (int test = 0; test < n_tests; ++ test)
        {
                const auto rowsA = rng_size();
                const auto colsA = rng_size();
                const auto rowsB = colsA;
                const auto colsB = rng_size();

                matrix_t A(rowsA, colsA);
                matrix_t B(rowsB, colsB);
                matrix_t C(rowsA, colsB);
                tensor::set_random(rng_value, A, B, C);

                cl::Buffer Abuffer = theocl.make_buffer(A, CL_MEM_READ_WRITE);
                cl::Buffer Bbuffer = theocl.make_buffer(B, CL_MEM_READ_WRITE);
                cl::Buffer Cbuffer = theocl.make_buffer(C, CL_MEM_READ_ONLY);

                kernel.setArg(0, Abuffer);
                kernel.setArg(1, static_cast<int>(colsA));
                kernel.setArg(2, Bbuffer);
                kernel.setArg(3, static_cast<int>(colsB));
                kernel.setArg(4, Cbuffer);

                NANO_CHECK(theocl.write(Abuffer, A) == CL_SUCCESS);
                NANO_CHECK(theocl.write(Bbuffer, B) == CL_SUCCESS);

                cl::CommandQueue& queue = theocl.command_queue();
                queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(size_t(rowsA), size_t(colsB)), cl::NullRange);
                queue.finish();

                NANO_CHECK(theocl.read(Cbuffer, C) == CL_SUCCESS);
                NANO_CHECK_EIGEN_CLOSE((A * B), C, nano::epsilon0<scalar_t>());
        }
}

NANO_END_MODULE()
