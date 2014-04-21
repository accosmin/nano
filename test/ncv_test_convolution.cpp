#include "nanocv.h"
#include "common/convolution.hpp"
#ifdef NANOCV_HAVE_OPENCL
        #include "opencl/opencl.h"
#endif

using namespace ncv;

#ifdef NANOCV_HAVE_OPENCL

const std::string conv_program_source = R"xxx(

#pragma OPENCL EXTENSION cl_amd_fp64 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable

__kernel void conv_kernel(
        __global const double* idata,
        __constant double* kdata, int krows, int kcols,
        __global double* odata)
{
        const int odims = get_global_size(0);
        const int ocols = get_global_size(1);
        const int orows = get_global_size(2);
        const int osize = orows * ocols;

        const int icols = ocols + kcols - 1;
        const int irows = orows + krows - 1;
        const int isize = irows * icols;

        const int o = get_global_id(0);
        const int x = get_global_id(1);
        const int y = get_global_id(2);

        const int iidx_base = o * isize + y * icols + x;

        double sum = 0;
        for (int r = 0, kidx = 0; r < krows; r ++)
        {
                int iidx = iidx_base + r * icols;
                for (int c = 0; c < kcols; c ++, kidx ++, iidx ++)
                {
                        sum += kdata[kidx] * idata[iidx];
                }
        }

        odata[o * osize + y * ocols + x] = sum;
}

)xxx";

#endif

ncv::thread_pool_t pool;
const size_t tests = 16;

void init_matrix(int rows, int cols, matrix_t& matrix)
{
        matrix.resize(rows, cols);
        matrix.setRandom();
        matrix /= rows;
}

void init_matrices(int rows, int cols, int count, matrices_t& matrices)
{
	matrices.resize(count);
	for (int i = 0; i < count; i ++)
	{
		init_matrix(rows, cols, matrices[i]);
	}
}

void zero_matrices(matrices_t& matrices)
{
        for (size_t i = 0; i < matrices.size(); i ++)
        {
                matrices[i].setZero();
        }
}

scalar_t sum_matrices(matrices_t& matrices)
{
        scalar_t sum = 0;
        for (size_t i = 0; i < matrices.size(); i ++)
        {
                sum += matrices[i].sum();
        }
        return sum;
}

template <typename top>
scalar_t test_conv2D_1cpu(top op, const char* name, const matrices_t& idatas, const matrix_t& kdata, matrices_t& odatas)
{
        ncv::stats_t<double, size_t> proc_stats;

        // run multiple tests
        for (size_t t = 0; t < tests; t ++)
        {
                zero_matrices(odatas);

                const ncv::timer_t timer;

                for (size_t i = 0; i < idatas.size(); i ++)
                {
                        op(idatas[i], kdata, odatas[i]);
                }

                proc_stats(timer.miliseconds());
        }

        const size_t milis = static_cast<size_t>(proc_stats.avg());
        std::cout << name << "= " << text::resize(text::to_string(milis), 4, align::right) << "ms  ";

        return sum_matrices(odatas);
}

template <typename top>
scalar_t test_conv2D_xcpu(top op, const char* name, const matrices_t& idatas, const matrix_t& kdata,  matrices_t& odatas)
{
        ncv::stats_t<double, size_t> proc_stats;

        // run multiple tests
        for (size_t t = 0; t < tests; t ++)
        {
                zero_matrices(odatas);

                const ncv::timer_t timer;

                ncv::thread_loop(idatas.size(), [&] (size_t i)
                {
                        op(idatas[i], kdata, odatas[i]);
                }, pool);

                proc_stats(timer.miliseconds());
        }

        const size_t milis = static_cast<size_t>(proc_stats.avg());
        std::cout << name << "= " << text::resize(text::to_string(milis), 4, align::right) << "ms  ";

        return sum_matrices(odatas);
}

#ifdef NANOCV_HAVE_OPENCL

scalar_t test_conv2D_gpu(const char* name, const matrices_t& idatas, const matrix_t& kdata, matrices_t& odatas, size_t tsend)
{
        ocl::manager_t& theocl = ocl::manager_t::instance();

        const cl::CommandQueue queue = theocl.make_command_queue();
        const cl::Program program = theocl.make_program_from_text(conv_program_source);
        cl::Kernel kernel = theocl.make_kernel(program, "conv_kernel");

        const int irows = static_cast<int>(idatas[0].rows());
        const int icols = static_cast<int>(idatas[0].cols());
        const int isize = irows * icols;
        const int krows = static_cast<int>(kdata.rows());
        const int kcols = static_cast<int>(kdata.cols());
        const int orows = static_cast<int>(odatas[0].rows());
        const int ocols = static_cast<int>(odatas[0].cols());
        const int osize = orows * ocols;

        scalars_t sidata(tsend * isize);
        scalars_t sodata(tsend * osize);

        const size_t mem_idata = idatas[0].size() * sizeof(scalar_t) * tsend;
        const size_t mem_kdata = kdata.size() * sizeof(scalar_t);
        const size_t mem_odata = odatas[0].size() * sizeof(scalar_t) * tsend;

        // create buffers once
        const cl::Buffer ibuffer = theocl.make_buffer(mem_idata, CL_MEM_READ_ONLY);
        const cl::Buffer kbuffer = theocl.make_buffer(mem_kdata, CL_MEM_READ_ONLY);
        const cl::Buffer obuffer = theocl.make_buffer(mem_odata, CL_MEM_WRITE_ONLY);

        // setup kernel buffers once
        kernel.setArg(0, ibuffer);
        kernel.setArg(1, kbuffer);
        kernel.setArg(2, sizeof(int), (void*)&krows);
        kernel.setArg(3, sizeof(int), (void*)&kcols);
        kernel.setArg(4, obuffer);

        // transfer constants
        queue.enqueueWriteBuffer(kbuffer, CL_TRUE, 0, mem_kdata, kdata.data());

        ncv::stats_t<double, size_t> proc_stats;

        // run multiple tests
        for (size_t t = 0; t < tests; t ++)
        {
                zero_matrices(odatas);

//                const ncv::timer_t timer;

                size_t micros = 0;
                for (size_t i = 0; i < idatas.size(); i += tsend)
                {
                        for (size_t it = 0; it < tsend; it ++)
                        {
                                const matrix_t& idata = idatas[i + it];
                                std::copy(idata.data(), idata.data() + idata.size(), sidata.data() + (it * isize));
                        }

                        const ncv::timer_t timer;

                        // I - send inputs to gpu
                        queue.enqueueWriteBuffer(ibuffer, CL_TRUE, 0, mem_idata, sidata.data());

                        // II - gpu processing
                        queue.enqueueNDRangeKernel(kernel, cl::NullRange,
                                                   cl::NDRange(tsend, ocols, orows),
                                                   cl::NDRange(1, ocols, orows));
                        queue.finish();

                        // III - read results from gpu
                        queue.enqueueReadBuffer(obuffer, CL_TRUE, 0, mem_odata, sodata.data());

                        micros += timer.microseconds();

                        for (size_t it = 0; it < tsend; it ++)
                        {
                                matrix_t& odata = odatas[i + it];
                                std::copy(sodata.data() + (it * osize), sodata.data() + (it * osize + osize), odata.data());
                        }
                }

                proc_stats(micros / 1000);

//                proc_stats(timer.miliseconds());
        }

        const size_t milis = static_cast<size_t>(proc_stats.avg());
        std::cout << name << "= " << text::resize(text::to_string(milis), 4, align::right) << "ms  ";

        return sum_matrices(odatas);
}

#endif

void test(int isize, int ksize, int n_samples)
{
        const int osize = isize - ksize + 1;

        matrices_t idatas, odatas;
        matrix_t kdata;

        init_matrices(isize, isize, n_samples, idatas);
        init_matrices(osize, osize, n_samples, odatas);
        init_matrix(ksize, ksize, kdata);

        std::cout << "(" << isize << "x" << isize << " @ " << ksize << "x" << ksize << "): ";
        const scalar_t sum1cpu = test_conv2D_1cpu(ncv::math::conv<matrix_t>, "conv(1CPU)", idatas, kdata, odatas);
        const scalar_t sumxcpu = test_conv2D_xcpu(ncv::math::conv<matrix_t>, "conv(xCPU)", idatas, kdata, odatas);
#ifdef NANOCV_HAVE_OPENCL
        const scalar_t sumg8dot = test_conv2D_gpu("dot(8GPU)", idatas, kdata, odatas, 8);
        const scalar_t sumg16dot = test_conv2D_gpu("dot(16GPU)", idatas, kdata, odatas, 16);
        const scalar_t sumg32dot = test_conv2D_gpu("dot(32GPU)", idatas, kdata, odatas, 32);
        const scalar_t sumg64dot = test_conv2D_gpu("dot(64GPU)", idatas, kdata, odatas, 64);
        const scalar_t sumg128dot = test_conv2D_gpu("dot(128GPU)", idatas, kdata, odatas, 128);
        const scalar_t sumg256dot = test_conv2D_gpu("dot(256GPU)", idatas, kdata, odatas, 256);
        const scalar_t sumg1024dot = test_conv2D_gpu("dot(1024GPU)", idatas, kdata, odatas, 1024);
#endif
        std::cout << std::endl;

        const scalar_t eps = 1e-12;//std::numeric_limits<scalar_t>::epsilon();
        scalar_t diff = 0.0;
        if ((diff = std::fabs(sum1cpu - sum1cpu)) > eps) { std::cout << "conv(1CPU) FAILED (diff = " << diff << ")!" << std::endl; }
        if ((diff = std::fabs(sumxcpu - sum1cpu)) > eps) { std::cout << "conv(xCPU) FAILED (diff = " << diff << ")!" << std::endl; }
#ifdef NANOCV_HAVE_OPENCL
        if ((diff = std::fabs(sumg8dot - sum1eib)) > eps) { std::cout << "dot(8GPU) FAILED (diff = " << diff << ")!" << std::endl; }
        if ((diff = std::fabs(sumg16dot - sum1eib)) > eps) { std::cout << "dot(16GPU) FAILED (diff = " << diff << ")!" << std::endl; }
        if ((diff = std::fabs(sumg32dot - sum1eib)) > eps) { std::cout << "dot(32GPU) FAILED (diff = " << diff << ")!" << std::endl; }
        if ((diff = std::fabs(sumg64dot - sum1eib)) > eps) { std::cout << "dot(64GPU) FAILED (diff = " << diff << ")!" << std::endl; }
        if ((diff = std::fabs(sumg128dot - sum1eib)) > eps) { std::cout << "dot(128GPU) FAILED (diff = " << diff << ")!" << std::endl; }
        if ((diff = std::fabs(sumg256dot - sum1eib)) > eps) { std::cout << "dot(256GPU) FAILED (diff = " << diff << ")!" << std::endl; }
        if ((diff = std::fabs(sumg1024dot - sum1eib)) > eps) { std::cout << "dot(1024GPU) FAILED (diff = " << diff << ")!" << std::endl; }
#endif
}

int main(int argc, char* argv[])
{
#ifdef NANOCV_HAVE_OPENCL
        if (!ocl::manager_t::instance().valid())
        {
                exit(EXIT_FAILURE);
        }
#endif

        static const int min_isize = 24;
        static const int max_isize = 48;
        static const int min_ksize = 5;
        static const int max_ksize = 13;
        static const int n_samples = 4 * 1024;

#ifdef NANOCV_HAVE_OPENCL
        try
#endif
        {
                for (int isize = min_isize; isize <= max_isize; isize += 4)
                {
                        for (int ksize = min_ksize; ksize <= max_ksize; ksize ++)
                        {
                                test(isize, ksize, n_samples);
                        }
                        std::cout << std::endl;
                }
        }

#ifdef NANOCV_HAVE_OPENCL
        catch (cl::Error e)
        {
                log_error() << "OpenCL fatal error: <" << e.what() << "> (" << ocl::error_string(e.err()) << ")!";
        }
#endif

	return EXIT_SUCCESS;
}

