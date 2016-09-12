#include "logger.h"
#include "tensor.h"
#include "measure.hpp"
#include "text/table.h"
#include "math/random.hpp"
#include "opencl/manager.h"
#include "tensor/numeric.hpp"
#include "text/table_row_mark.h"
#include <iomanip>
#include <iostream>

namespace
{
        using namespace nano;

        opencl_manager_t theocl;

        nano::random_t<scalar_t> rng(scalar_t(-1e-3), scalar_t(+1e-3));
        const size_t trials = 16;

        auto measure_read(const tensor_size_t dims)
        {
                vector_t x(dims);
                tensor::set_random(rng, x);

                cl::Buffer buffer = theocl.make_buffer(x, CL_MEM_READ_WRITE);
                theocl.write(buffer, x);

                volatile scalar_t z = 0;
                const auto duration = nano::measure_robustly_psec([&] ()
                {
                        theocl.read(buffer, x);
                        ++ z;
                }, trials);

                return nano::gflops(dims, duration);
        }

        auto measure_write(const tensor_size_t dims)
        {
                vector_t x(dims);
                tensor::set_random(rng, x);

                cl::Buffer buffer = theocl.make_buffer(x, CL_MEM_READ_WRITE);

                volatile scalar_t z = 0;
                const auto duration = nano::measure_robustly_psec([&] ()
                {
                        theocl.write(buffer, x);
                        ++ z;
                }, trials);

                return nano::gflops(dims, duration);
        }
}

int main(int, char* [])
{
        using namespace nano;

        try
        {
                theocl.init();
                theocl.select(CL_DEVICE_TYPE_GPU);

                const auto min_dims = tensor_size_t(8);
                const auto max_dims = tensor_size_t(1024);
                const auto foreach_dims = [&] (const auto& op)
                {
                        for (tensor_size_t dims = min_dims; dims <= max_dims; dims *= 2)
                        {
                                op(dims);
                        }
                };
                const auto fillrow = [&] (auto&& row, const auto& op)
                {
                        foreach_dims([&] (const auto dims)
                        {
                                row << op(dims);
                        });
                };

                table_t table("operation\\dimensions [GFLOPS]");
                foreach_dims([&] (const auto dims) { table.header() << to_string(dims); });

                fillrow(table.append("read"), measure_read);
                fillrow(table.append("write"), measure_write);

                table.mark(nano::make_table_mark_maximum_percentage_cols<scalar_t>(10));
                table.print(std::cout);
        }

        catch (cl::Error& e)
        {
                log_error() << "OpenCL fatal error: <" << e.what() << "> (" << error_string(e.err()) << ")!";
                return EXIT_FAILURE;
        }

        return EXIT_SUCCESS;
}

