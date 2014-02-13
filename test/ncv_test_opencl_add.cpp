#include "ncv.h"
#include "opencl/opencl.h"

const std::string program_source = R"xxx(

#pragma OPENCL EXTENSION cl_amd_fp64 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable

__kernel void test_kernel(
       __global const double* a,
       __global const double* b,
       __global double* result)
{
       const int gid = get_global_id(0);
       result[gid] = a[gid] + b[gid] * b[gid];
}

)xxx";

template
<
        typename tscalar
>
tscalar cpu_op(tscalar a, tscalar b)
{
        return a + b * b;
}

template
<
        typename tvector
>
bool check(const tvector& a, const tvector& b, const tvector& c, const char* error_message)
{
        const auto eps = std::numeric_limits<typename tvector::Scalar>::epsilon();

        for (auto i = 0; i < a.size(); i ++)
        {
                const auto diff = std::fabs(c(i) - cpu_op(a(i), b(i)));
                if (diff > eps)
                {
                        ncv::log_error() << error_message << " (diff = " << diff << ")";
                        return false;
                }
        }

        return true;
}

int main(int argc, char *argv[])
{
        using namespace ncv;

        try
        {
                if (!ocl::manager_t::instance().valid())
                {
                        exit(EXIT_FAILURE);
                }

                const cl::Context& context = ocl::manager_t::instance().context();
                const cl::CommandQueue& queue = ocl::manager_t::instance().queue();

                const cl::Program program = ocl::manager_t::instance().program_from_text(program_source);

                cl::Kernel kernel = cl::Kernel(program, "test_kernel");

                const size_t tests = 32;
                const size_t minsize = 1024;
                const size_t maxsize = 4 * 1024 * 1024;

                // try various data sizes
                for (size_t size = minsize; size <= maxsize; size *= 4)
                {
                        ncv::stats_t<double, size_t> send_stats;        // send inputs to gpu
                        ncv::stats_t<double, size_t> proc_stats;        // gpu processing
                        ncv::stats_t<double, size_t> read_stats;        // read results from gpu
                        ncv::stats_t<double, size_t> scpu_stats;        // cpu processing (single core)
                        ncv::stats_t<double, size_t> mcpu_stats;        // cpu processing (multiple cores)

                        ncv::tensor::vector_types_t<double>::tvector a; a.resize(size);
                        ncv::tensor::vector_types_t<double>::tvector b; b.resize(size);
                        ncv::tensor::vector_types_t<double>::tvector c; c.resize(size);

                        const size_t array_size = size * sizeof(double);

                        // create buffers once
                        cl::Buffer cl_a = cl::Buffer(context, CL_MEM_READ_ONLY, array_size, NULL);
                        cl::Buffer cl_b = cl::Buffer(context, CL_MEM_READ_ONLY, array_size, NULL);
                        cl::Buffer cl_c = cl::Buffer(context, CL_MEM_WRITE_ONLY, array_size, NULL);

                        // setup kernel buffers once
                        kernel.setArg(0, cl_a);
                        kernel.setArg(1, cl_b);
                        kernel.setArg(2, cl_c);
                        queue.finish();

                        // run multiple tests
                        for (size_t test = 0; test < tests; test ++)
                        {
                                ncv::timer_t timer;

                                for (size_t i = 0; i < size; i ++)
                                {
                                        a(i) = (1.0f + i) / (size + 0.0f) + test;
                                        b(i) = (2.0f + i) / (size + 0.0f) - test;
                                        c(i) = 0.0f;
                                }

                                // I - send inputs to gpu
                                timer.start();
                                {
                                        cl::Event event;
                                        queue.enqueueWriteBuffer(cl_a, CL_FALSE, 0, array_size, a.data(), NULL, &event);
                                        queue.enqueueWriteBuffer(cl_b, CL_FALSE, 0, array_size, b.data(), NULL, &event);
                                        queue.finish();
                                }
                                send_stats(timer.microseconds());

                                // II - gpu processing
                                timer.start();
                                {
                                        cl::Event event;
                                        queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(size), cl::NullRange, NULL, &event);
                                        queue.finish();
                                }
                                proc_stats(timer.microseconds());

                                // III - read results from gpu
                                timer.start();
                                {
                                        cl::Event event;
                                        queue.enqueueReadBuffer(cl_c, CL_TRUE, 0, array_size, c.data(), NULL, &event);
                                }
                                read_stats(timer.microseconds());

                                check(a, b, c, "GPU processing failed: incorrect result!");

                                // IV - single-threaded cpu processing
                                timer.start();
                                {
                                        for (size_t i = 0; i < size; i ++)
                                        {
                                                c(i) = cpu_op(a(i), b(i));
                                        }
                                }
                                scpu_stats(timer.microseconds());

                                check(a, b, c, "sCPU processing failed: incorrect result!");

                                // V - multi-threaded cpu processing
                                timer.start();
                                {
                                        ncv::thread_loop(size, [&] (size_t i)
                                        {
                                                c(i) = cpu_op(a(i), b(i));
                                        });
                                }
                                mcpu_stats(timer.microseconds());

                                check(a, b, c, "mCPU processing failed: incorrect result!");
                        }

                        // results
                        log_info() << "SIZE [" << text::resize(text::to_string(size / 1024), 4, align::right) << "K]"
                                   << ": sendGPU= " << text::resize(text::to_string(send_stats.avg()), 14, align::right) << "us"
                                   << ", procGPU= " << text::resize(text::to_string(proc_stats.avg()), 14, align::right) << "us"
                                   << ", readGPU= " << text::resize(text::to_string(read_stats.avg()), 14, align::right) << "us"
                                   << ", 1CPU= " << text::resize(text::to_string(scpu_stats.avg()), 14, align::right) << "us"
                                   << ", xCPU= " << text::resize(text::to_string(mcpu_stats.avg()), 14, align::right) << "us";
                }
        }

        catch (cl::Error e)
        {
                log_error() << "OpenCL fatal error: <" << e.what() << "> (" << ocl::error_string(e.err()) << ")!";
        }

        // OK
        ncv::log_info() << ncv::done;
        return EXIT_SUCCESS;
}
