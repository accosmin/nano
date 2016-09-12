#include "utest.hpp"
#include "tensor.h"
#include "math/random.hpp"
#include "opencl/opencl.h"
#include "opencl/kernels.h"
#include "math/epsilon.hpp"
#include "tensor/numeric.hpp"

using namespace nano;

auto rng_size = nano::make_rng<tensor_size_t>(1, 1024);
auto rng_value = nano::make_rng<scalar_t>(-1, +1);

NANO_BEGIN_MODULE(test_opencl)

NANO_CASE(addv)
{
        opencl_manager_t theocl;
        NANO_REQUIRE_NOTHROW(theocl.init());
        NANO_REQUIRE_NOTHROW(theocl.select(CL_DEVICE_TYPE_GPU));

        cl::Program program;
        NANO_REQUIRE_NOTHROW(theocl.make_program_from_text(opencl_kernel_addv()));

        cl::Kernel kernel;
        NANO_REQUIRE_NOTHROW(theocl.make_kernel(program, "addv"));

        for (int test = 0; test < 11; ++ test)
        {
                const auto dims = rng_size();

                vector_t x(dims), y(dims), z(dims);
                tensor::set_random(rng_value, x, y, z);

                cl::Buffer xbuffer = theocl.make_buffer(tensor_size(x), CL_MEM_READ_WRITE);
                cl::Buffer ybuffer = theocl.make_buffer(tensor_size(y), CL_MEM_READ_WRITE);
                cl::Buffer zbuffer = theocl.make_buffer(tensor_size(z), CL_MEM_READ_ONLY);

                kernel.setArg(0, xbuffer);
                kernel.setArg(1, ybuffer);
                kernel.setArg(2, zbuffer);

                cl::CommandQueue& queue = theocl.command_queue();
                queue.enqueueWriteBuffer(xbuffer, CL_FALSE, 0, tensor_size(x), x.data());
                queue.enqueueWriteBuffer(ybuffer, CL_FALSE, 0, tensor_size(y), y.data());
                queue.finish();

                queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(size_t(dims)), cl::NullRange);
                queue.finish();

                queue.enqueueReadBuffer(zbuffer, CL_TRUE, 0, tensor_size(z), z.data());
                NANO_CHECK_EIGEN_CLOSE((x + y), z, nano::epsilon0<scalar_t>());
        }
}

NANO_END_MODULE()
