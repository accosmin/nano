#include "logger.h"
#include "tensor.h"
#include "measure.hpp"
#include "text/table.h"
#include "opencl/opencl.h"
#include "math/epsilon.hpp"

int main(int, char* [])
{
        using namespace nano;

        try
        {
                ocl::manager_t theocl;

                const auto context = theocl.make_context();
                const auto queue = theocl.make_command_queue(context);

                const size_t trials = 16;
                const size_t minsize = 32;
                const size_t maxsize = 32 * 1024;

                table_t table("vector size \\ GPU operation");
                table.header() << "write [ns]" << "read [ns]";

                // try various data sizes
                for (size_t size = minsize; size <= maxsize; size *= 2)
                {
                        vector_t a(size);
                        vector_t b(size);

                        const auto array_size = size * sizeof(scalar_t);
                        const auto abuffer = theocl.make_buffer(context, array_size, CL_MEM_READ_WRITE);

                        a.setRandom();
                        b.setRandom();

                        // write to GPU
                        const auto write_ns = nano::measure_robustly_nsec([&] ()
                        {
                                queue.enqueueWriteBuffer(abuffer, CL_TRUE, 0, array_size, a.data());
                        }, trials).count();

                        // read from GPU
                        const auto read_ns = nano::measure_robustly_nsec([&] ()
                        {
                                queue.enqueueReadBuffer(abuffer, CL_TRUE, 0, array_size, b.data());
                        }, trials).count();

                        // check buffers
                        if ((a - b).lpNorm<Eigen::Infinity>() > nano::epsilon0<scalar_t>())
                        {
                                log_error() << "copying to/from GPU failed!";
                        }

                        table.append(size) << write_ns << read_ns;
                }

                table.print(std::cout);
        }

        catch (cl::Error& e)
        {
                log_error() << "OpenCL fatal error: <" << e.what() << "> (" << ocl::error_string(e.err()) << ")!";
                return EXIT_FAILURE;
        }

        return EXIT_SUCCESS;
}

