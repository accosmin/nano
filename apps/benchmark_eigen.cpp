#include "tensor.h"
#include "measure.hpp"
#include "text/table.h"
#include "text/cmdline.h"
#include "math/random.hpp"
#include "tensor/numeric.hpp"
#include "text/table_row_mark.h"
#ifdef NANO_WITH_OPENCL
#include "logger.h"
#include "opencl/kernels.h"
#include "opencl/manager.h"
#endif
#include <iostream>

namespace
{
        using namespace nano;

        auto rng_value = nano::make_rng(scalar_t(-1e-3), scalar_t(+1e-3));
        const size_t trials = 16;

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

#ifdef NANO_WITH_OPENCL
        opencl_manager_t theocl;
        cl::CommandQueue& queue = theocl.command_queue();

        cl::Kernel kernel_vpc;
        cl::Kernel kernel_vpv;
        cl::Kernel kernel_vcpvc;

        cl::Kernel kernel_mv;
        cl::Kernel kernel_mvpc;
        cl::Kernel kernel_mvpv;

        cl::Kernel kernel_mm;
#endif

        auto measure_vpc(const tensor_size_t dims)
        {
                auto x = make_vector(dims);
                auto c = make_scalar();
                auto z = make_vector(dims);

                const auto duration = nano::measure_robustly_psec([&] ()
                {
                        z = x.array() + c;
                }, trials);

                return nano::gflops(dims, duration);
        }

        auto measure_vpv(const tensor_size_t dims)
        {
                auto x = make_vector(dims);
                auto y = make_vector(dims);
                auto z = make_vector(dims);

                const auto duration = nano::measure_robustly_psec([&] ()
                {
                        z = x + y;
                }, trials);

                return nano::gflops(dims, duration);
        }

        auto measure_vcpvc(const tensor_size_t dims)
        {
                auto x = make_vector(dims); auto a = make_scalar();
                auto y = make_vector(dims); auto b = make_scalar();
                auto z = make_vector(dims);

                const auto duration = nano::measure_robustly_psec([&] ()
                {
                        z = x.array() * a + y.array() * b;
                }, trials);

                return nano::gflops(3 * dims, duration);
        }

        auto measure_mv(const tensor_size_t dims)
        {
                auto A = make_matrix(dims, dims);
                auto x = make_vector(dims);
                auto z = make_vector(dims);

                const auto duration = nano::measure_robustly_psec([&] ()
                {
                        z = A * x;
                }, trials);

                return nano::gflops(dims * dims, duration);
        }

        auto measure_mvpc(const tensor_size_t dims)
        {
                auto A = make_matrix(dims, dims);
                auto x = make_vector(dims);
                auto c = make_scalar();
                auto z = make_vector(dims);

                const auto duration = nano::measure_robustly_psec([&] ()
                {
                        z = (A * x).array() + c;
                }, trials);

                return nano::gflops(dims * dims + dims, duration);
        }

        auto measure_mvpv(const tensor_size_t dims)
        {
                auto A = make_matrix(dims, dims);
                auto x = make_vector(dims);
                auto y = make_vector(dims);
                auto z = make_vector(dims);

                const auto duration = nano::measure_robustly_psec([&] ()
                {
                        z = A * x + y;
                }, trials);

                return nano::gflops(dims * dims + dims, duration);
        }

        auto measure_mm(const tensor_size_t dims)
        {
                auto A = make_matrix(dims, dims);
                auto B = make_matrix(dims, dims);
                auto Z = make_matrix(dims, dims);

                const auto duration = nano::measure_robustly_psec([&] ()
                {
                        Z = A * B;
                }, trials);

                return nano::gflops(dims * dims * dims, duration);
        }
}

int main(int, const char* [])
{
        using namespace nano;

#ifdef NANO_WITH_OPENCL
        try
        {
        // initialize OpenCL context
        theocl.init();
        theocl.select(CL_DEVICE_TYPE_GPU);

        // create supported kernels
        cl::Program program = theocl.make_program_from_text(opencl_kernels());

        kernel_vpc = theocl.make_kernel(program, "vpc");
        kernel_vpv = theocl.make_kernel(program, "vpv");
        kernel_vcpvc = theocl.make_kernel(program, "vcpvc");

        kernel_mv = theocl.make_kernel(program, "mv");
        kernel_mvpc = theocl.make_kernel(program, "mvpc");
        kernel_mvpv = theocl.make_kernel(program, "mvpv");

        kernel_mm = theocl.make_kernel(program, "mm");
#endif

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

        fillrow(table.append("vpc: z = x + c"), measure_vpc);
        fillrow(table.append("vpv: z = x + y"), measure_vpv);
        fillrow(table.append("vcpvc: z = a * x + b * y"), measure_vcpvc);
        fillrow(table.append("mv: z = A * x"), measure_mv);
        fillrow(table.append("mvpc: z = A * x + c"), measure_mvpc);
        fillrow(table.append("mvpv: z = A * x + v"), measure_mvpv);
        fillrow(table.append("mm: Z = A * B"), measure_mm);

        table.mark(nano::make_table_mark_maximum_percentage_cols<scalar_t>(10));
        table.print(std::cout);

#ifdef NANO_WITH_OPENCL
        }
        catch (cl::Error& e)
        {
                log_error() << "OpenCL fatal error: <" << e.what() << "> (" << error_string(e.err()) << ")!";
                return EXIT_FAILURE;
        }
#endif

        return EXIT_SUCCESS;
}

