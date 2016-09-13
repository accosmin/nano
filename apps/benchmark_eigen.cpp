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

#ifdef NANO_WITH_OPENCL
        auto measure_vpc_ocl(const tensor_size_t dims)
        {
                auto x = make_vector(dims);
                auto c = make_scalar();
                auto z = make_vector(dims);

                cl::Buffer xbuffer = theocl.make_buffer(x, CL_MEM_READ_WRITE);
                cl::Buffer zbuffer = theocl.make_buffer(z, CL_MEM_READ_ONLY);

                nano::set_args(kernel_vpc, xbuffer, c, zbuffer);
                theocl.write(xbuffer, x);

                const auto duration = nano::measure_robustly_psec([&] ()
                {
                        queue.enqueueNDRangeKernel(kernel_vpc, cl::NullRange, cl::NDRange(size_t(dims)), cl::NullRange);
                        queue.finish();
                }, trials);

                theocl.read(zbuffer, z);

                return nano::gflops(dims, duration);
        }
#endif

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

#ifdef NANO_WITH_OPENCL
        auto measure_vpv_ocl(const tensor_size_t dims)
        {
                auto x = make_vector(dims);
                auto y = make_vector(dims);
                auto z = make_vector(dims);

                cl::Buffer xbuffer = theocl.make_buffer(x, CL_MEM_READ_WRITE);
                cl::Buffer ybuffer = theocl.make_buffer(y, CL_MEM_READ_WRITE);
                cl::Buffer zbuffer = theocl.make_buffer(z, CL_MEM_READ_ONLY);

                nano::set_args(kernel_vpv, xbuffer, ybuffer, zbuffer);
                theocl.write(xbuffer, x);
                theocl.write(ybuffer, y);

                const auto duration = nano::measure_robustly_psec([&] ()
                {
                        queue.enqueueNDRangeKernel(kernel_vpv, cl::NullRange, cl::NDRange(size_t(dims)), cl::NullRange);
                        queue.finish();
                }, trials);

                theocl.read(zbuffer, z);

                return nano::gflops(dims, duration);
        }
#endif
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

#ifdef NANO_WITH_OPENCL
        auto measure_vcpvc_ocl(const tensor_size_t dims)
        {
                auto x = make_vector(dims); auto a = make_scalar();
                auto y = make_vector(dims); auto b = make_scalar();
                auto z = make_vector(dims);

                cl::Buffer xbuffer = theocl.make_buffer(x, CL_MEM_READ_WRITE);
                cl::Buffer ybuffer = theocl.make_buffer(y, CL_MEM_READ_WRITE);
                cl::Buffer zbuffer = theocl.make_buffer(z, CL_MEM_READ_ONLY);

                nano::set_args(kernel_vcpvc, xbuffer, a, ybuffer, b, zbuffer);
                theocl.write(xbuffer, x);
                theocl.write(ybuffer, y);

                const auto duration = nano::measure_robustly_psec([&] ()
                {
                        queue.enqueueNDRangeKernel(kernel_vcpvc, cl::NullRange, cl::NDRange(size_t(dims)), cl::NullRange);
                        queue.finish();
                }, trials);

                theocl.read(zbuffer, z);

                return nano::gflops(3 * dims, duration);
        }
#endif

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

#ifdef NANO_WITH_OPENCL
        auto measure_mv_ocl(const tensor_size_t dims)
        {
                auto A = make_matrix(dims, dims);
                auto x = make_vector(dims);
                auto z = make_vector(dims);

                cl::Buffer Abuffer = theocl.make_buffer(A, CL_MEM_READ_WRITE);
                cl::Buffer xbuffer = theocl.make_buffer(x, CL_MEM_READ_WRITE);
                cl::Buffer zbuffer = theocl.make_buffer(z, CL_MEM_READ_ONLY);

                nano::set_args(kernel_mv, Abuffer, int(dims), xbuffer, zbuffer);
                theocl.write(Abuffer, A);
                theocl.write(xbuffer, x);

                const auto duration = nano::measure_robustly_psec([&] ()
                {
                        queue.enqueueNDRangeKernel(kernel_mv, cl::NullRange, cl::NDRange(size_t(dims)), cl::NullRange);
                        queue.finish();
                }, trials);

                theocl.read(zbuffer, z);

                return nano::gflops(dims * dims, duration);
        }
#endif

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

#ifdef NANO_WITH_OPENCL
        auto measure_mvpc_ocl(const tensor_size_t dims)
        {
                auto A = make_matrix(dims, dims);
                auto x = make_vector(dims);
                auto c = make_scalar();
                auto z = make_vector(dims);

                cl::Buffer Abuffer = theocl.make_buffer(A, CL_MEM_READ_WRITE);
                cl::Buffer xbuffer = theocl.make_buffer(x, CL_MEM_READ_WRITE);
                cl::Buffer zbuffer = theocl.make_buffer(z, CL_MEM_READ_ONLY);

                nano::set_args(kernel_mvpc, Abuffer, int(dims), xbuffer, c, zbuffer);
                theocl.write(Abuffer, A);
                theocl.write(xbuffer, x);

                const auto duration = nano::measure_robustly_psec([&] ()
                {
                        queue.enqueueNDRangeKernel(kernel_mvpc, cl::NullRange, cl::NDRange(size_t(dims)), cl::NullRange);
                        queue.finish();
                }, trials);

                theocl.read(zbuffer, z);

                return nano::gflops(dims * dims + dims, duration);
        }
#endif

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

#ifdef NANO_WITH_OPENCL
        auto measure_mvpv_ocl(const tensor_size_t dims)
        {
                auto A = make_matrix(dims, dims);
                auto x = make_vector(dims);
                auto y = make_vector(dims);
                auto z = make_vector(dims);

                cl::Buffer Abuffer = theocl.make_buffer(A, CL_MEM_READ_WRITE);
                cl::Buffer xbuffer = theocl.make_buffer(x, CL_MEM_READ_WRITE);
                cl::Buffer ybuffer = theocl.make_buffer(y, CL_MEM_READ_WRITE);
                cl::Buffer zbuffer = theocl.make_buffer(z, CL_MEM_READ_ONLY);

                nano::set_args(kernel_mvpv, Abuffer, (int)dims, xbuffer, ybuffer, zbuffer);
                theocl.write(Abuffer, A);
                theocl.write(xbuffer, x);
                theocl.write(ybuffer, y);

                const auto duration = nano::measure_robustly_psec([&] ()
                {
                        queue.enqueueNDRangeKernel(kernel_mvpv, cl::NullRange, cl::NDRange(size_t(dims)), cl::NullRange);
                        queue.finish();
                }, trials);

                theocl.read(zbuffer, z);

                return nano::gflops(dims * dims + dims, duration);
        }
#endif

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

#ifdef NANO_WITH_OPENCL
        auto measure_mm_ocl(const tensor_size_t dims)
        {
                auto A = make_matrix(dims, dims);
                auto B = make_matrix(dims, dims);
                auto Z = make_matrix(dims, dims);

                cl::Buffer Abuffer = theocl.make_buffer(A, CL_MEM_READ_WRITE);
                cl::Buffer Bbuffer = theocl.make_buffer(B, CL_MEM_READ_WRITE);
                cl::Buffer Zbuffer = theocl.make_buffer(Z, CL_MEM_READ_ONLY);

                nano::set_args(kernel_mm, Abuffer, (int)dims, Bbuffer, (int)dims, Zbuffer);
                theocl.write(Abuffer, A);
                theocl.write(Bbuffer, B);

                const auto duration = nano::measure_robustly_psec([&] ()
                {
                        queue.enqueueNDRangeKernel(kernel_mm, cl::NullRange, cl::NDRange(size_t(dims), size_t(dims)), cl::NullRange);
                        queue.finish();
                }, trials);

                theocl.read(Zbuffer, Z);

                return nano::gflops(dims * dims * dims, duration);
        }
#endif
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

        table_t table("operation");
        table.header() << "platform";
        foreach_dims([&] (const auto dims) { table.header() << (to_string(dims) + " [GFLOPS]"); });

        fillrow(table.append("z = x + c") << "CPU", measure_vpc);
#ifdef NANO_WITH_OPENCL
        fillrow(table.append("z = x + c") << "OpenCL", measure_vpc_ocl);
#endif
        fillrow(table.append("z = x + y") << "CPU", measure_vpv);
#ifdef NANO_WITH_OPENCL
        fillrow(table.append("z = x + y") << "OpenCL", measure_vpv_ocl);
#endif
        fillrow(table.append("z = a * x + b * y") << "CPU", measure_vcpvc);
#ifdef NANO_WITH_OPENCL
        fillrow(table.append("z = a * x + b * y") << "OpenCL", measure_vcpvc_ocl);
#endif
        fillrow(table.append("z = A * x") << "CPU", measure_mv);
#ifdef NANO_WITH_OPENCL
        fillrow(table.append("z = A * x") << "OpenCL", measure_mv_ocl);
#endif
        fillrow(table.append("z = A * x + c") << "CPU", measure_mvpc);
#ifdef NANO_WITH_OPENCL
        fillrow(table.append("z = A * x + c") << "OpenCL", measure_mvpc_ocl);
#endif
        fillrow(table.append("z = A * x + y") << "CPU", measure_mvpv);
#ifdef NANO_WITH_OPENCL
        fillrow(table.append("z = A * x + y") << "OpenCL", measure_mvpv_ocl);
#endif
        fillrow(table.append("Z = A * B") << "CPU", measure_mm);
#ifdef NANO_WITH_OPENCL
        fillrow(table.append("z = A * B") << "OpenCL", measure_mm_ocl);
#endif

        //table.mark(nano::make_table_mark_maximum_percentage_cols<scalar_t>(10));
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

