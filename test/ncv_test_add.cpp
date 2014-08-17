#include "nanocv.h"
#ifdef NANOCV_HAVE_OPENCL
#include "opencl/opencl.h"
#endif
#ifdef NANOCV_HAVE_CUDA
#include "cuda/cuda.h"
#endif

const std::string program_source = R"xxx(

#pragma OPENCL EXTENSION cl_amd_fp64 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable

__kernel void add_kernel(
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
        typedef typename tvector::Scalar scalar_t;

        const auto eps = std::numeric_limits<scalar_t>::epsilon();

        ncv::stats_t<scalar_t> stats;
        for (auto i = 0; i < a.size(); i ++)
        {
                stats(std::fabs(c(i) - cpu_op(a(i), b(i))));
        }

        if (stats.max() > eps)
        {
                ncv::log_error() << error_message << " (diff = [" << stats.min() << ", " << stats.max()
                                 << "] ~ " << stats.avg() << ")";
                return false;
        }

        else
        {
                return true;
        }
}

int main(int argc, char *argv[])
{
        using namespace ncv;

#ifdef NANOCV_HAVE_OPENCL
        try
        {
                ocl::manager_t& theocl = ocl::manager_t::instance();

                if (!theocl.valid())
                {
                        exit(EXIT_FAILURE);
                }

                const cl::Context context = theocl.make_context();
                const cl::CommandQueue queue = theocl.make_command_queue(context);
                const cl::Program program = theocl.make_program_from_text(context, program_source);
                cl::Kernel kernel = theocl.make_kernel(program, "add_kernel");
#endif

#ifdef NANOCV_HAVE_CUDA
                cuda::print_info();
#endif

                const size_t tests = 1000;
                const size_t minsize = 1024;
                const size_t maxsize = 1024 * 1024;

                thread_pool_t pool;

                // try various data sizes
                for (size_t size = minsize; size <= maxsize; size *= 2)
                {
#if defined(NANOCV_HAVE_OPENCL) || defined(NANOCV_HAVE_CUDA)
                        ncv::stats_t<double, size_t> send_stats;        // send inputs to gpu
                        ncv::stats_t<double, size_t> proc_stats;        // gpu processing
                        ncv::stats_t<double, size_t> read_stats;        // read results from gpu
#endif
                        ncv::stats_t<double, size_t> scpu_stats;        // cpu processing (single core)                        
                        ncv::stats_t<double, size_t> mcpu_stats;        // cpu processing (multiple cores)

                        ncv::tensor::vector_types_t<double>::tvector a; a.resize(size);
                        ncv::tensor::vector_types_t<double>::tvector b; b.resize(size);
                        ncv::tensor::vector_types_t<double>::tvector c; c.resize(size);

#ifdef NANOCV_HAVE_OPENCL
                        const size_t array_size = size * sizeof(double);

                        // create buffers once
                        const cl::Buffer abuffer = theocl.make_buffer(context, array_size, CL_MEM_READ_ONLY);
                        const cl::Buffer bbuffer = theocl.make_buffer(context, array_size, CL_MEM_READ_ONLY);
                        const cl::Buffer cbuffer = theocl.make_buffer(context, array_size, CL_MEM_WRITE_ONLY);

                        // setup kernel buffers once
                        kernel.setArg(0, abuffer);
                        kernel.setArg(1, bbuffer);
                        kernel.setArg(2, cbuffer);
#endif

#ifdef NANOCV_HAVE_CUDA
                        cuda::device_buffer_t d_abuffer(size);
                        cuda::device_buffer_t d_bbuffer(size);
                        cuda::device_buffer_t d_cbuffer(size);
#endif

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

#ifdef NANOCV_HAVE_OPENCL
                                // GPU - copy to device
                                timer.start();
                                {
                                        queue.enqueueWriteBuffer(abuffer, CL_TRUE, 0, array_size, a.data());
                                        queue.enqueueWriteBuffer(bbuffer, CL_TRUE, 0, array_size, b.data());
                                }
                                send_stats(timer.microseconds());

                                // GPU - process
                                timer.start();
                                {
                                        queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(size), cl::NullRange);
                                        queue.finish();
                                }
                                proc_stats(timer.microseconds());

                                // GPU - copy from device
                                timer.start();
                                {
                                        queue.enqueueReadBuffer(cbuffer, CL_TRUE, 0, array_size, c.data());
                                }
                                read_stats(timer.microseconds());

                                check(a, b, c, "GPU processing failed: incorrect result!");
#endif

#ifdef NANOCV_HAVE_CUDA
                                // GPU - copy to device
                                timer.start();
                                {
                                        d_abuffer.copyToDevice(a.data());
                                        d_bbuffer.copyToDevice(b.data());
                                }
                                send_stats(timer.microseconds());

                                // GPU - process
                                timer.start();
                                {
                                        cuda::addbsquared(d_abuffer, d_bbuffer, d_cbuffer);
                                }
                                proc_stats(timer.microseconds());

                                // GPU - copy from device
                                timer.start();
                                {
                                        d_cbuffer.copyFromDevice(c.data());
                                }
                                read_stats(timer.microseconds());

                                check(a, b, c, "GPU processing failed: incorrect result!");
#endif

                                // CPU - single-threaded
                                timer.start();
                                {
                                        for (size_t i = 0; i < size; i ++)
                                        {
                                                c(i) = cpu_op(a(i), b(i));
                                        }
                                }
                                scpu_stats(timer.microseconds());

                                check(a, b, c, "sCPU processing failed: incorrect result!");

                                // CPU - multi-threaded
                                timer.start();
                                {
                                        ncv::thread_loopi(size, pool, [&] (size_t i)
                                        {
                                                c(i) = cpu_op(a(i), b(i));
                                        });
                                }
                                mcpu_stats(timer.microseconds());

                                check(a, b, c, "mCPU processing failed: incorrect result!");
                        }

#if defined(NANOCV_HAVE_OPENCL) || defined(NANOCV_HAVE_CUDA)
                        const size_t time_send = static_cast<size_t>(0.5 + send_stats.sum() / 1000);
                        const size_t time_proc = static_cast<size_t>(0.5 + proc_stats.sum() / 1000);
                        const size_t time_read = static_cast<size_t>(0.5 + read_stats.sum() / 1000);
#endif
                        const size_t time_scpu = static_cast<size_t>(0.5 + scpu_stats.sum() / 1000);
                        const size_t time_mcpu = static_cast<size_t>(0.5 + mcpu_stats.sum() / 1000);

                        // results
                        log_info() << "SIZE [" << text::resize(text::to_string(size / 1024), 4, align::right) << "K] x "
                                   << "TIMES [" << text::resize(text::to_string(tests), 0, align::right) << "]: "
#if defined(NANOCV_HAVE_OPENCL) || defined(NANOCV_HAVE_CUDA)
                                   << "sendGPU= " << text::resize(text::to_string(time_send), 8, align::right) << "ms, "
                                   << "procGPU= " << text::resize(text::to_string(time_proc), 8, align::right) << "ms, "
                                   << "readGPU= " << text::resize(text::to_string(time_read), 8, align::right) << "ms, "
#endif
                                   << "singCPU= " << text::resize(text::to_string(time_scpu), 8, align::right) << "ms, "
                                   << "multCPU= " << text::resize(text::to_string(time_mcpu), 8, align::right) << "ms";
                }

#ifdef NANOCV_HAVE_OPENCL
        }

        catch (cl::Error& e)
        {
                log_error() << "OpenCL fatal error: <" << e.what() << "> (" << ocl::error_string(e.err()) << ")!";
        }
#endif

        // OK
        ncv::log_info() << ncv::done;
        return EXIT_SUCCESS;
}
