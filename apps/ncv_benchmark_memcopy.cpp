#include "libnanocv/types.h"
#include "libnanocv/math/abs.hpp"
#include "libnanocv/util/timer.h"
#include "libnanocv/util/logger.h"
#include "libnanocv/util/stats.hpp"
#include "libnanocv/thread/parallel.hpp"
#ifdef NANOCV_HAVE_OPENCL
#include "opencl/opencl.h"
#endif
#ifdef NANOCV_HAVE_CUDA
#include "cuda/cuda.h"
#include "cuda/vector.hpp"
#endif

template
<
        typename tvector
>
static bool check(const tvector& a, const tvector& b, const char* error_message)
{
        typedef typename tvector::Scalar scalar_t;

        const auto eps = std::numeric_limits<scalar_t>::epsilon();

        ncv::stats_t<scalar_t> stats;
        for (auto i = 0; i < a.size(); i ++)
        {
                stats(ncv::math::abs(b(i) - a(i)));
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
                        ncv::stats_t<double, size_t> read_stats;        // read results from gpu
#endif
                        ncv::stats_t<double, size_t> scpu_stats;        // cpu processing (single core)                        
                        ncv::stats_t<double, size_t> mcpu_stats;        // cpu processing (multiple cores)

                        ncv::tensor::vector_types_t<double>::tvector a; a.resize(size);
                        ncv::tensor::vector_types_t<double>::tvector b; b.resize(size);

#ifdef NANOCV_HAVE_OPENCL
                        const size_t array_size = size * sizeof(double);

                        const cl::Buffer abuffer = theocl.make_buffer(context, array_size, CL_MEM_READ_WRITE);

#endif

#ifdef NANOCV_HAVE_CUDA
                        cuda::vector_t<double> d_abuffer(size);
#endif

                        // run multiple tests
                        for (size_t test = 0; test < tests; test ++)
                        {
                                ncv::timer_t timer;
                                
                                a.setRandom();
                                b.setRandom();

#ifdef NANOCV_HAVE_OPENCL
                                // GPU - copy to device
                                timer.start();
                                {
                                        queue.enqueueWriteBuffer(abuffer, CL_TRUE, 0, array_size, a.data());
                                }
                                send_stats(timer.microseconds());

                                // GPU - copy from device
                                timer.start();
                                {
                                        queue.enqueueReadBuffer(abuffer, CL_TRUE, 0, array_size, b.data());
                                }
                                read_stats(timer.microseconds());

                                check(a, b, "GPU processing failed: incorrect result!");
#endif

#ifdef NANOCV_HAVE_CUDA
                                // GPU - copy to device
                                timer.start();
                                {
                                        d_abuffer.to_device(a.data());
                                }
                                send_stats(timer.microseconds());

                                // GPU - copy from device
                                timer.start();
                                {
                                        d_abuffer.from_device(b.data());
                                }
                                read_stats(timer.microseconds());

                                check(a, b, "GPU processing failed: incorrect result!");
#endif

                                // CPU - single-threaded
                                timer.start();
                                {
                                        for (size_t i = 0; i < size; i ++)
                                        {
                                                b(i) = a(i);
                                        }
                                }
                                scpu_stats(timer.microseconds());

                                // CPU - multi-threaded
                                timer.start();
                                {
                                        ncv::thread_loopi(size, pool, [&] (size_t i)
                                        {
                                                b(i) = a(i);
                                        });
                                }
                                mcpu_stats(timer.microseconds());
                        }

#if defined(NANOCV_HAVE_OPENCL) || defined(NANOCV_HAVE_CUDA)
                        const size_t time_send = static_cast<size_t>(0.5 + send_stats.sum() / 1000);
                        const size_t time_read = static_cast<size_t>(0.5 + read_stats.sum() / 1000);
#endif
                        const size_t time_scpu = static_cast<size_t>(0.5 + scpu_stats.sum() / 1000);
                        const size_t time_mcpu = static_cast<size_t>(0.5 + mcpu_stats.sum() / 1000);

                        // results
                        log_info() << "SIZE [" << text::resize(text::to_string(size / 1024), 4, align::right) << "K] x "
                                   << "TIMES [" << text::resize(text::to_string(tests), 0, align::right) << "]: "
#if defined(NANOCV_HAVE_OPENCL) || defined(NANOCV_HAVE_CUDA)
                                   << "sendGPU= " << text::resize(text::to_string(time_send), 8, align::right) << "ms, "
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
