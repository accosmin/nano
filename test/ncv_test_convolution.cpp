#include "nanocv.h"
#include "common/conv2d.hpp"
#ifdef NANOCV_HAVE_OPENCL
#include "opencl/opencl.h"
#endif
#ifdef NANOCV_HAVE_CUDA
#include "cuda/cuda.h"
#include "cuda/conv2d.h"
#endif

using namespace ncv;

ncv::thread_pool_t pool;
const size_t tests = 16;

template
<
        typename tmatrix
>
void init_matrix(int rows, int cols, tmatrix& matrix)
{
        matrix.resize(rows, cols);
        matrix.setRandom();
        matrix /= rows;
}

template
<
        typename tmatrix
>
void init_matrices(int rows, int cols, int count, std::vector<tmatrix>& matrices)
{
	matrices.resize(count);
	for (int i = 0; i < count; i ++)
	{
		init_matrix(rows, cols, matrices[i]);
	}
}

template
<
        typename tmatrix
>
void zero_matrices(std::vector<tmatrix>& matrices)
{
        for (size_t i = 0; i < matrices.size(); i ++)
        {
                matrices[i].setZero();
        }
}

template
<
        typename tmatrix,
        typename tscalar = typename tmatrix::Scalar
>
tscalar sum_matrices(std::vector<tmatrix>& matrices)
{
        tscalar sum = 0;
        for (size_t i = 0; i < matrices.size(); i ++)
        {
                sum += matrices[i].sum();
        }
        return sum;
}

template
<
        typename top,
        typename tmatrix,
        typename tscalar = typename tmatrix::Scalar
>
tscalar test_cpu(
        top op, const char* name,
        const std::vector<tmatrix>&, const tmatrix&, std::vector<tmatrix>& odatas)
{
        ncv::stats_t<double, size_t> proc_stats;
        
        // run multiple tests
        for (size_t t = 0; t < tests; t ++)
        {
                zero_matrices(odatas);
                
                const ncv::timer_t timer;
                op();
                
                proc_stats(timer.miliseconds());
        }
        
        const size_t milis = static_cast<size_t>(proc_stats.avg());
        std::cout << name << "= " << text::resize(text::to_string(milis), 4, align::right) << "ms  ";
        
        return sum_matrices(odatas);
}

template <typename top>
scalar_t test_1cpu(top op, const char* name, const matrices_t& idatas, const matrix_t& kdata, matrices_t& odatas)
{
        return test_cpu([&] ()
        {
                for (size_t i = 0; i < idatas.size(); i ++)
                {
                        op(idatas[i], kdata, odatas[i]);
                }
        }, name, idatas, kdata, odatas);
}

template <typename top>
scalar_t test_xcpu(top op, const char* name, const matrices_t& idatas, const matrix_t& kdata,  matrices_t& odatas)
{
        return test_cpu([&] ()
        {
                ncv::thread_loopi(idatas.size(), pool, [&] (size_t i)
                {
                        op(idatas[i], kdata, odatas[i]);
                });
        }, name, idatas, kdata, odatas);
}

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

scalar_t test_gpu(const char* name, const matrices_t& idatas, const matrix_t& kdata, matrices_t& odatas)
{
        ocl::manager_t& theocl = ocl::manager_t::instance();

        const cl::Context context = theocl.make_context();
        const cl::CommandQueue queue = theocl.make_command_queue(context);
        const cl::Program program = theocl.make_program_from_text(context, conv_program_source);
        cl::Kernel kernel = theocl.make_kernel(program, "conv_kernel");

        const int krows = static_cast<int>(kdata.rows());
        const int kcols = static_cast<int>(kdata.cols());
        const int orows = static_cast<int>(odatas[0].rows());
        const int ocols = static_cast<int>(odatas[0].cols());

        const size_t mem_idata = idatas[0].size() * sizeof(scalar_t);
        const size_t mem_kdata = kdata.size() * sizeof(scalar_t);
        const size_t mem_odata = odatas[0].size() * sizeof(scalar_t);

        // create buffers once
        const cl::Buffer ibuffer = theocl.make_buffer(context, mem_idata, CL_MEM_READ_ONLY);
        const cl::Buffer kbuffer = theocl.make_buffer(context, mem_kdata, CL_MEM_READ_ONLY);
        const cl::Buffer obuffer = theocl.make_buffer(context, mem_odata, CL_MEM_WRITE_ONLY);

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

                size_t micros = 0;
                for (size_t i = 0; i < idatas.size(); i ++)
                {
                        const ncv::timer_t timer;

                        queue.enqueueWriteBuffer(ibuffer, CL_TRUE, 0, mem_idata, idatas[i].data());

                        queue.enqueueNDRangeKernel(kernel, cl::NullRange,
                                                   cl::NDRange(1, ocols, orows),
                                                   cl::NDRange(1, ocols, orows));
                        queue.finish();

                        queue.enqueueReadBuffer(obuffer, CL_TRUE, 0, mem_odata, odatas[i].data());

                        micros += timer.microseconds();
                }

                proc_stats(micros / 1000);
        }

        const size_t milis = static_cast<size_t>(proc_stats.avg());
        std::cout << name << "= " << text::resize(text::to_string(milis), 4, align::right) << "ms  ";

        return sum_matrices(odatas);
}

#endif

#ifdef NANOCV_HAVE_CUDA

template
<
        typename top
>
scalar_t test_gpu(top op, const char* name, const matrices_t& idatas, const matrix_t& kdata, matrices_t& odatas)
{
        const int irows = static_cast<int>(idatas[0].rows());
        const int icols = static_cast<int>(idatas[0].cols());
        const int krows = static_cast<int>(kdata.rows());
        const int kcols = static_cast<int>(kdata.cols());
        const int orows = static_cast<int>(odatas[0].rows());
        const int ocols = static_cast<int>(odatas[0].cols());

        cuda::dmatrix_t d_idata(irows, icols);
        cuda::dmatrix_t d_kdata(krows, kcols);
        cuda::dmatrix_t d_odata(orows, ocols);

        // transfer constants
        d_kdata.copyToDevice(kdata.data());

        ncv::stats_t<double, size_t> proc_stats;

        // run multiple tests
        for (size_t t = 0; t < tests; t ++)
        {
                zero_matrices(odatas);

                size_t micros = 0;
                for (size_t i = 0; i < idatas.size(); i ++)
                {
                        const ncv::timer_t timer;

                        d_idata.copyToDevice(idatas[i].data());

                        op(d_idata, d_kdata, d_odata);

                        d_odata.copyFromDevice(odatas[i].data());

                        micros += timer.microseconds();
                }

                proc_stats(micros / 1000);
        }

        const size_t milis = static_cast<size_t>(proc_stats.avg());
        std::cout << name << "= " << text::resize(text::to_string(milis), 4, align::right) << "ms  ";

        return sum_matrices(odatas);
}

#endif

void check(scalar_t result, scalar_t baseline, const char* name)
{
        const scalar_t eps = 1e-12;//std::numeric_limits<scalar_t>::epsilon();
        const scalar_t err = std::fabs(result - baseline);

        if (err > eps)
        {
                std::cout << name << " FAILED (diff = " << err << ")!" << std::endl;
        }
}

void test_conv2d(int isize, int ksize, int n_samples)
{
        const int osize = isize - ksize + 1;

        matrices_t idatas, odatas;
        matrix_t kdata;

        init_matrices(isize, isize, n_samples, idatas);
        init_matrices(osize, osize, n_samples, odatas);
        init_matrix(ksize, ksize, kdata);
        
        const string_t header = (boost::format("(%1%x%2%@%3%x%4%): ") % isize % isize % ksize % ksize).str();
        std::cout << text::resize(header, 16);
        
        const scalar_t conve1cpu  = test_1cpu(ncv::conv_eig<matrix_t>, "eig(1CPU)", idatas, kdata, odatas);
        const scalar_t convexcpu  = test_xcpu(ncv::conv_eig<matrix_t>, "eig(xCPU)", idatas, kdata, odatas);
        const scalar_t convd1cpu  = test_1cpu(ncv::conv_dot<matrix_t>, "dot(1CPU)", idatas, kdata, odatas);
        const scalar_t convdxcpu  = test_xcpu(ncv::conv_dot<matrix_t>, "dot(xCPU)", idatas, kdata, odatas);
#if defined(NANOCV_HAVE_OPENCL)
        const scalar_t convgpu    = test_gpu("conv2d(GPU)", idatas, kdata, odatas);
#elif defined(NANOCV_HAVE_CUDA)
        const scalar_t convgpu    = test_gpu(std::bind(cuda::conv2d, _1, _2, _3, 0), "conv2d(GPU)", idatas, kdata, odatas);
#endif
        std::cout << std::endl;

        check(conve1cpu, conve1cpu, "conve(1CPU)");
        check(convexcpu, conve1cpu, "conve(xCPU)");
        check(convd1cpu, conve1cpu, "convd(1CPU)");
        check(convdxcpu, conve1cpu, "convd(xCPU)");
#if defined(NANOCV_HAVE_OPENCL) || defined(NANOCV_HAVE_CUDA)
        check(convgpu  , conve1cpu, "conv2d(GPU)");
#endif
}

void test_iconv2d(int isize, int ksize, int n_samples)
{
        const int osize = isize - ksize + 1;

        matrices_t idatas, odatas;
        matrix_t kdata;

        init_matrices(isize, isize, n_samples, idatas);
        init_matrices(osize, osize, n_samples, odatas);
        init_matrix(ksize, ksize, kdata);

        const string_t header = (boost::format("(%1%x%2%@%3%x%4%): ") % isize % isize % ksize % ksize).str();
        std::cout << text::resize(header, 16);

        const scalar_t iconve1cpu = test_1cpu(ncv::iconv_eig<matrix_t>, "ieig(1CPU)", odatas, kdata, idatas);
        const scalar_t iconvexcpu = test_xcpu(ncv::iconv_eig<matrix_t>, "ieig(xCPU)", odatas, kdata, idatas);
        const scalar_t iconvm1cpu = test_1cpu(ncv::iconv_mad<matrix_t>, "imad(1CPU)", odatas, kdata, idatas);
        const scalar_t iconvmxcpu = test_xcpu(ncv::iconv_mad<matrix_t>, "imad(xCPU)", odatas, kdata, idatas);
#ifdef NANOCV_HAVE_CUDA
        const scalar_t iconvgpu   = test_gpu(std::bind(cuda::iconv2d, _1, _2, _3, 0), "iconv2d(GPU)", odatas, kdata, idatas);
#endif
        std::cout << std::endl;

        check(iconve1cpu, iconve1cpu, "iconve(1CPU)");
        check(iconvexcpu, iconve1cpu, "iconve(xCPU)");
        check(iconvm1cpu, iconve1cpu, "iconvm(1CPU)");
        check(iconvmxcpu, iconve1cpu, "iconvm(xCPU)");
#ifdef NANOCV_HAVE_CUDA
        check(iconvgpu  , iconve1cpu, "iconv2d(GPU)");
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

#ifdef NANOCV_HAVE_CUDA
        cuda::print_info();
#endif

        static const int min_isize = 24;
        static const int max_isize = 48;
        static const int min_ksize = 5;
        static const int n_samples = 4 * 1024;

#ifdef NANOCV_HAVE_OPENCL
        try
#endif
        {
                for (int isize = min_isize; isize <= max_isize; isize += 4)
                {
                        for (int ksize = min_ksize; ksize <= isize - min_ksize; ksize += 2)
                        {
                                test_conv2d(isize, ksize, n_samples);
                        }
                        std::cout << std::endl;

                        for (int ksize = min_ksize; ksize <= isize - min_ksize; ksize += 2)
                        {
                                test_iconv2d(isize, ksize, n_samples);
                        }
                        std::cout << std::endl;
                }
        }

#ifdef NANOCV_HAVE_OPENCL
        catch (cl::Error& e)
        {
                log_error() << "OpenCL fatal error: <" << e.what() << "> (" << ocl::error_string(e.err()) << ")!";
        }
#endif

	return EXIT_SUCCESS;
}

