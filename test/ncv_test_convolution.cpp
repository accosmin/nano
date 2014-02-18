#include "ncv.h"
#include "common/convolution.hpp"
#include "opencl/opencl.h"

using namespace ncv;

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

        double sum = 0;
        for (int r = 0, kidx = 0; r < krows; r ++)
        {
                int iidx = o * isize + (y + r) * icols + x;
                for (int c = 0; c < kcols; c ++, kidx ++, iidx ++)
                {
                        sum += kdata[kidx] * idata[iidx];
                }
        }

        odata[o * osize + y * ocols + x] = sum;
}

)xxx";

ncv::thread_pool_t pool;
const size_t tests = 4;

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

scalar_t test_conv2D_gpu(const char* name, const matrices_t& idatas, const matrix_t& kdata, matrices_t& odatas)
{
        const cl::Context& context = ocl::manager_t::instance().context();
        const cl::CommandQueue& queue = ocl::manager_t::instance().queue();

        const cl::Program program = ocl::manager_t::instance().program_from_text(conv_program_source);
        cl::Kernel kernel = cl::Kernel(program, "conv_kernel");

        const int irows = static_cast<int>(idatas[0].rows());
        const int icols = static_cast<int>(idatas[0].cols());
        const int isize = irows * icols;
        const int krows = static_cast<int>(kdata.rows());
        const int kcols = static_cast<int>(kdata.cols());
        const int orows = static_cast<int>(odatas[0].rows());
        const int ocols = static_cast<int>(odatas[0].cols());
        const int osize = orows * ocols;
        const int tsend = 10;

        scalars_t sidata(tsend * isize);
        scalars_t sodata(tsend * osize);

        const size_t mem_idata = idatas[0].size() * sizeof(scalar_t) * tsend;
        const size_t mem_kdata = kdata.size() * sizeof(scalar_t);
        const size_t mem_odata = odatas[0].size() * sizeof(scalar_t) * tsend;

        // create buffers once
        cl::Buffer cl_idata = cl::Buffer(context, CL_MEM_READ_ONLY, mem_idata, NULL);
        cl::Buffer cl_kdata = cl::Buffer(context, CL_MEM_READ_ONLY, mem_kdata, NULL);
        cl::Buffer cl_odata = cl::Buffer(context, CL_MEM_WRITE_ONLY, mem_odata, NULL);

        // setup kernel buffers once
        kernel.setArg(0, cl_idata);
        kernel.setArg(1, cl_kdata);
        kernel.setArg(2, sizeof(int), (void*)&krows);
        kernel.setArg(3, sizeof(int), (void*)&kcols);
        kernel.setArg(4, cl_odata);
        queue.finish();

        // transfer constants
        cl::Event event;
        queue.enqueueWriteBuffer(cl_kdata, CL_FALSE, 0, mem_kdata, kdata.data(), NULL, &event);
        queue.finish();

        ncv::stats_t<double, size_t> proc_stats;

        // run multiple tests
        for (size_t t = 0; t < tests; t ++)
        {
                zero_matrices(odatas);

                ncv::timer_t timer;

                for (size_t i = 0; i < idatas.size(); i += tsend)
                {
                        for (size_t it = 0; it < tsend; it ++)
                        {
                                const matrix_t& idata = idatas[i + it];
                                std::copy(idata.data(), idata.data() + idata.size(), sidata.data() + (it * isize));
                        }

                        // I - send inputs to gpu
                        cl::Event event;
                        queue.enqueueWriteBuffer(cl_idata, CL_FALSE, 0, mem_idata, sidata.data(), NULL, &event);
                        queue.finish();

                        // II - gpu processing
                        queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(tsend, ocols, orows), cl::NullRange, NULL, &event);
                        queue.finish();

                        // III - read results from gpu
                        queue.enqueueReadBuffer(cl_odata, CL_TRUE, 0, mem_odata, sodata.data(), NULL, &event);

                        for (size_t it = 0; it < tsend; it ++)
                        {
                                matrix_t& odata = odatas[i + it];
                                std::copy(sodata.data() + (it * osize), sodata.data() + (it * osize + osize), odata.data());
                        }
                }

                proc_stats(timer.miliseconds());
        }

        const size_t milis = static_cast<size_t>(proc_stats.avg());
        std::cout << name << "= " << text::resize(text::to_string(milis), 4, align::right) << "ms  ";

        return sum_matrices(odatas);
}

void test(int isize, int ksize, int n_samples)
{
        const int osize = isize - ksize + 1;

        matrices_t idatas, odatas;
        matrix_t kdata;

        init_matrices(isize, isize, n_samples, idatas);
        init_matrices(osize, osize, n_samples, odatas);
        init_matrix(ksize, ksize, kdata);

        std::cout << "mix (isize = " << isize << ", ksize = " << ksize << "): \t";
        const scalar_t sum1eib = test_conv2D_1cpu(ncv::math::conv_eib<matrix_t>, "eib(1CPU)", idatas, kdata, odatas);
        const scalar_t sumxeib = test_conv2D_xcpu(ncv::math::conv_eib<matrix_t>, "eib(xCPU)", idatas, kdata, odatas);
        const scalar_t sum1dot = test_conv2D_1cpu(ncv::math::conv_dot<matrix_t>, "dot(1CPU)", idatas, kdata, odatas);
        const scalar_t sumxdot = test_conv2D_xcpu(ncv::math::conv_dot<matrix_t>, "dot(xCPU)", idatas, kdata, odatas);
        const scalar_t sumgdot = test_conv2D_gpu("dot(GPU)", idatas, kdata, odatas);
        std::cout << std::endl;

        const scalar_t eps = 1e-12;//std::numeric_limits<scalar_t>::epsilon();
        scalar_t diff = 0.0;
        if ((diff = std::fabs(sum1eib - sum1eib)) > eps) { std::cout << "eib(1CPU) FAILED (diff = " << diff << ")!" << std::endl; }
        if ((diff = std::fabs(sumxeib - sum1eib)) > eps) { std::cout << "eib(xCPU) FAILED (diff = " << diff << ")!" << std::endl; }
        if ((diff = std::fabs(sum1dot - sum1eib)) > eps) { std::cout << "dot(1CPU) FAILED (diff = " << diff << ")!" << std::endl; }
        if ((diff = std::fabs(sumxdot - sum1eib)) > eps) { std::cout << "dot(xCPU) FAILED (diff = " << diff << ")!" << std::endl; }
        if ((diff = std::fabs(sumgdot - sum1eib)) > eps) { std::cout << "dot( GPU) FAILED (diff = " << diff << ")!" << std::endl; }
}

int main(int argc, char* argv[])
{
        if (!ocl::manager_t::instance().valid())
        {
                exit(EXIT_FAILURE);
        }

        static const int min_isize = 24;
        static const int max_isize = 48;
        static const int min_ksize = 5;
        static const int max_ksize = 13;
        static const int n_samples = 10000;

        try
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

        catch (cl::Error e)
        {
                log_error() << "OpenCL fatal error: <" << e.what() << "> (" << ocl::error_string(e.err()) << ")!";
        }

	return EXIT_SUCCESS;
}

