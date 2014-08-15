#include "nanocv.h"
#include "opencl/opencl.h"

const std::string program_source = R"xxx(

#pragma OPENCL EXTENSION cl_amd_fp64 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable

__kernel void copy_kernel(
       __global const double* idata,
       __global double* odata)
{
       const int i = get_global_id(0);
       odata[i] = idata[i];
}

)xxx";

int main(int argc, char *argv[])
{
        using namespace ncv;

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
                cl::Kernel kernel = theocl.make_kernel(program, "copy_kernel");

                const size_t tests = 10000;
                const size_t minsize = 1024;
                const size_t maxsize = 64 * 1024;

                // try various data sizes
                for (size_t size = minsize; size <= maxsize; size *= 2)
                {
                        ncv::stats_t<scalar_t, size_t> send_stats;        // send data to gpu
                        ncv::stats_t<scalar_t, size_t> copy_stats;        // copy data inside gpu
                        ncv::stats_t<scalar_t, size_t> read_stats;        // read data from gpu
                        ncv::stats_t<scalar_t, size_t> ccpu_stats;        // cpu-based copy

                        ncv::tensor::vector_types_t<double>::tvector rdata; rdata.resize(size);
                        ncv::tensor::vector_types_t<double>::tvector wdata; wdata.resize(size);

                        const size_t array_size = size * sizeof(double);

                        // create buffers
                        const cl::Buffer rbuffer = theocl.make_buffer(context, array_size, CL_MEM_READ_ONLY);
                        const cl::Buffer wbuffer = theocl.make_buffer(context, array_size, CL_MEM_WRITE_ONLY);

                        // setup kernel buffers
                        kernel.setArg(0, rbuffer);
                        kernel.setArg(1, wbuffer);

                        // run multiple tests
                        for (size_t test = 0; test < tests; test ++)
                        {                                
                                rdata.setRandom();

                                ncv::timer_t timer;

                                timer.start();
                                {
                                        queue.enqueueWriteBuffer(rbuffer, CL_TRUE, 0, array_size, rdata.data());
                                }
                                send_stats(timer.microseconds());

                                timer.start();
                                {
                                        queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(size), cl::NullRange);
                                        queue.finish();
                                }
                                copy_stats(timer.microseconds());

                                timer.start();
                                {
                                        queue.enqueueReadBuffer(wbuffer, CL_TRUE, 0, array_size, wdata.data());
                                }
                                read_stats(timer.microseconds());

                                if (!std::equal(wdata.data(), wdata.data() + wdata.size(), rdata.data()))
                                {
                                        ncv::log_error() << "failed to copy data to & from GPU!";
                                        return EXIT_FAILURE;
                                }
                                
                                timer.start();
                                {
                                        wdata = rdata;
                                }
                                ccpu_stats(timer.microseconds());
                        }

                        const size_t time_send = static_cast<size_t>(0.5 + send_stats.sum() / 1000);
                        const size_t time_copy = static_cast<size_t>(0.5 + copy_stats.sum() / 1000);
                        const size_t time_read = static_cast<size_t>(0.5 + read_stats.sum() / 1000);
                        const size_t time_ccpu = static_cast<size_t>(0.5 + ccpu_stats.sum() / 1000);

                        // results
                        log_info() << "SIZE [" << text::resize(text::to_string(size / 1024), 2, align::right) << "K]"
                                   << ", TIMES [" << text::resize(text::to_string(tests), 2, align::right) << "]"
                                   << ": sendGPU= " << text::resize(text::to_string(time_send), 6, align::right) << "ms"
                                   << ", copyGPU= " << text::resize(text::to_string(time_copy), 6, align::right) << "ms"
                                   << ", readGPU= " << text::resize(text::to_string(time_read), 6, align::right) << "ms"
                                   << ", copyCPU= " << text::resize(text::to_string(time_ccpu), 6, align::right) << "ms";
                }
        }

        catch (cl::Error& e)
        {
                log_error() << "OpenCL fatal error: <" << e.what() << "> (" << ocl::error_string(e.err()) << ")!";
        }

        // OK
        ncv::log_info() << ncv::done;
        return EXIT_SUCCESS;
}
