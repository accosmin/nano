#include "ncv.h"
#include "opencl/opencl.h"

const char* program_source = "\n" \
"__kernel void add_kernel(                              \n" \
"       __global const float* a,                        \n" \
"       __global const float* b,                        \n" \
"       __global float* result)                         \n" \
"{                                                      \n" \
"       int gid = get_global_id(0);                     \n" \
"       result[gid] = a[gid] + b[gid];                  \n" \
"}                                                      \n" \
"__kernel void mul_kernel(                              \n" \
"       __global const float* a,                        \n" \
"       __global const float* b,                        \n" \
"       __global float* result)                         \n" \
"{                                                      \n" \
"       int gid = get_global_id(0);                     \n" \
"       result[gid] = a[gid] * b[gid];                  \n" \
"}                                                      \n" \
"\n";

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

//                cl::Kernel add_kernel = cl::Kernel(program, "add_kernel");
                cl::Kernel mul_kernel = cl::Kernel(program, "mul_kernel");

                const size_t tests = 16;
                const size_t minsize = 1024;
                const size_t maxsize = 64 * 1024 * 1024;

                // try various data sizes
                for (size_t size = minsize; size <= maxsize; size *= 4)
                {
                        ncv::stats_t<double, size_t> send_stats;        // send inputs to gpu
                        ncv::stats_t<double, size_t> proc_stats;        // gpu processing
                        ncv::stats_t<double, size_t> read_stats;        // read results from gpu
                        ncv::stats_t<double, size_t> scpu_stats;        // cpu processing (single core)
                        ncv::stats_t<double, size_t> mcpu_stats;        // cpu processing (multiple cores)

                        std::vector<float> a(size);
                        std::vector<float> b(size);
                        std::vector<float> c(size);

                        const size_t array_size = size * sizeof(float);

                        for (size_t i = 0; i < size; i ++)
                        {
                                a[i] = 1.0f * i;
                                b[i] = 1.0f * i;
                                c[i] = 0.0f;
                        }

                        // run multiple tests
                        for (size_t test = 0; test < tests; test ++)
                        {
                                ncv::timer_t timer;

                                cl::Buffer cl_a = cl::Buffer(context, CL_MEM_READ_ONLY, array_size, NULL);
                                cl::Buffer cl_b = cl::Buffer(context, CL_MEM_READ_ONLY, array_size, NULL);
                                cl::Buffer cl_c = cl::Buffer(context, CL_MEM_WRITE_ONLY, array_size, NULL);

                                // I - send inputs to gpu
                                timer.start();
                                {
                                        cl::Event event;
                                        queue.enqueueWriteBuffer(cl_a, CL_TRUE, 0, array_size, a.data(), NULL, &event);
                                        queue.enqueueWriteBuffer(cl_b, CL_TRUE, 0, array_size, b.data(), NULL, &event);
                                        queue.enqueueWriteBuffer(cl_c, CL_TRUE, 0, array_size, c.data(), NULL, &event);

                                        mul_kernel.setArg(0, cl_a);
                                        mul_kernel.setArg(1, cl_b);
                                        mul_kernel.setArg(2, cl_c);
                                        queue.finish();
                                }
                                send_stats(timer.miliseconds());

                                // II - gpu processing
                                timer.start();
                                {
                                        cl::Event event;
                                        queue.enqueueNDRangeKernel(mul_kernel, cl::NullRange, cl::NDRange(size), cl::NullRange, NULL, &event);
                                        queue.finish();
                                }
                                proc_stats(timer.miliseconds());

                                // III - read results from gpu
                                timer.start();
                                {
                                        cl::Event event;
                                        queue.enqueueReadBuffer(cl_c, CL_TRUE, 0, array_size, c.data(), NULL, &event);
                                }
                                read_stats(timer.miliseconds());

                                // check results
                                for (size_t i = 0; i < size; i ++)
                                {
                                        if (std::fabs(c[i] - (a[i] * b[i])) > 0.00001f)
                                        {
                                                log_error() << "GPU processing failed: incorrect result!";
                                                break;
                                        }
                                }

                                // IV - single-threaded cpu processing
                                timer.start();
                                {
                                        for (size_t i = 0; i < size; i ++)
                                        {
                                                c[i] = a[i] * b[i];
                                        }
                                }
                                scpu_stats(timer.miliseconds());

                                // TODO: cpu processing times
                        }

                        // results
                        log_info() << "SIZE [" << (size / 1024) << "K]"
                                   << ": send - " << send_stats.avg() << " +/- " << send_stats.stdev() << " ms"
                                   << ", proc - " << proc_stats.avg() << " +/- " << proc_stats.stdev() << " ms"
                                   << ", read - " << read_stats.avg() << " +/- " << read_stats.stdev() << " ms"
                                   << ", scpu - " << scpu_stats.avg() << " +/- " << scpu_stats.stdev() << " ms";
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
