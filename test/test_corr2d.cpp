#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE "test_corr2d"

#include <boost/test/unit_test.hpp>
#include "nanocv/tensor.h"
#include "nanocv/logger.h"
#include "nanocv/corr2d.hpp"
#include "nanocv/math/close.hpp"
#include "nanocv/math/epsilon.hpp"
#ifdef NANOCV_WITH_OPENCL
#include "nanocv/opencl/opencl.h"
#endif
#ifdef NANOCV_WITH_CUDA
#include "cuda/cuda.h"
#include "cuda/conv2d.h"
#endif

namespace test
{
        using namespace ncv;

        template
        <
                typename top,
                typename tmatrix,
                typename tscalar = typename tmatrix::Scalar
        >
        tscalar test_cpu(top op, const tmatrix& idata, const tmatrix& kdata, tmatrix& odata)
        {
                odata.setZero();

                op(idata, kdata, odata);

                return odata.sum();
        }

        #ifdef NANOCV_WITH_OPENCL

        const std::string conv_program_source = R"xxx(

        #pragma OPENCL EXTENSION cl_amd_fp64 : enable
        #pragma OPENCL EXTENSION cl_khr_fp64 : enable

        __kernel void corr_kernel(
                __global const double* odata,
                __constant double* kdata, int krows, int kcols,
                __global double* idata)
        {
                const int c = get_global_id(0);
                const int r = get_global_id(1);

                const int icols = get_global_size(0);
                const int irows = get_global_size(1);

                const int orows = irows - krows + 1;
                const int ocols = icols - kcols + 1;

                const int krmin = max(0,     r - orows + 1);
                const int krmax = min(krows, r + 1);

                const int kcmin = max(0,     c - ocols + 1);
                const int kcmax = min(kcols, c + 1);

                double sum = 0;
                for (int kr = krmin; kr < krmax; kr ++)
                {
                        for (int kc = kcmin; kc < kcmax; kc ++)
                        {
                                sum += odata[(r - kr) * ocols + (c - kc)] * kdata[kr * kcols + kc];
                        }
                }

                idata[r * icols + c] = sum;
        }

        )xxx";

        template
        <
                typename tmatrix,
                typename tscalar = typename tmatrix::Scalar
        >
        tscalar test_gpu(
                const char* kernel_name, const char* name,
                const tmatrix& idata, const tmatrix& kdata, tmatrix& odata)
        {
                ocl::manager_t& theocl = ocl::get_manager();

                const cl::Context context = theocl.make_context();
                const cl::CommandQueue queue = theocl.make_command_queue(context);
                const cl::Program program = theocl.make_program_from_text(context, conv_program_source);
                cl::Kernel kernel = theocl.make_kernel(program, kernel_name);

                const int krows = static_cast<int>(kdata.rows());
                const int kcols = static_cast<int>(kdata.cols());
                const int orows = static_cast<int>(odata.rows());
                const int ocols = static_cast<int>(odata.cols());

                const size_t mem_idata = idata.size() * sizeof(tscalar);
                const size_t mem_kdata = kdata.size() * sizeof(tscalar);
                const size_t mem_odata = odata.size() * sizeof(tscalar);

                // create buffers
                const cl::Buffer ibuffer = theocl.make_buffer(context, mem_idata, CL_MEM_READ_ONLY);
                const cl::Buffer kbuffer = theocl.make_buffer(context, mem_kdata, CL_MEM_READ_ONLY);
                const cl::Buffer obuffer = theocl.make_buffer(context, mem_odata, CL_MEM_WRITE_ONLY);

                // setup kernel buffers
                kernel.setArg(0, ibuffer);
                kernel.setArg(1, kbuffer);
                kernel.setArg(2, sizeof(int), (void*)&krows);
                kernel.setArg(3, sizeof(int), (void*)&kcols);
                kernel.setArg(4, obuffer);

                // transfer constants
                queue.enqueueWriteBuffer(kbuffer, CL_TRUE, 0, mem_kdata, kdata.data());

                // compute
                queue.enqueueWriteBuffer(ibuffer, CL_TRUE, 0, mem_idata, idata.data());
                queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(ocols, orows), cl::NDRange(ocols, orows));
                queue.finish();
                queue.enqueueReadBuffer(obuffer, CL_TRUE, 0, mem_odata, odata.data());

                return odata.sum();
        }

        #endif

        #ifdef NANOCV_WITH_CUDA

        template
        <
                typename top,
                typename tmatrix,
                typename tscalar = typename tmatrix::Scalar
        >
        tscalar test_gpu(
                top op, const char* name,
                const std::vector<tmatrix>& idatas, const tmatrix& kdata, std::vector<tmatrix>& odatas)
        {
                const int irows = static_cast<int>(idatas[0].rows());
                const int icols = static_cast<int>(idatas[0].cols());
                const int krows = static_cast<int>(kdata.rows());
                const int kcols = static_cast<int>(kdata.cols());
                const int orows = static_cast<int>(odatas[0].rows());
                const int ocols = static_cast<int>(odatas[0].cols());

                cuda::matrix_t<tscalar> d_idata(irows, icols);
                cuda::matrix_t<tscalar> d_kdata(krows, kcols);
                cuda::matrix_t<tscalar> d_odata(orows, ocols);

                ncv::stats_t<double, size_t> proc_stats;

                // run multiple tests
                for (size_t t = 0; t < tests; t ++)
                {
                        zero_matrices(odatas);

                        const ncv::timer_t timer;

                        d_kdata.to_device(kdata.data());

                        for (size_t i = 0; i < idatas.size(); i ++)
                        {
                                d_idata.to_device(idatas[i].data());

                                op(d_idata, d_kdata, d_odata, 0);

                                d_odata.from_device(odatas[i].data());
                        }

                        proc_stats(timer.miliseconds());
                }

                const size_t milis = static_cast<size_t>(proc_stats.min());
                std::cout << name << "= " << text::resize(text::to_string(milis), 4, align::right) << "ms  ";

                return sum_matrices(odatas);
        }

        #endif

        void test_corr2d(int isize, int ksize)
        {
                const int osize = isize - ksize + 1;

                matrix_t idata(isize, isize);
                matrix_t kdata(ksize, ksize);
                matrix_t odata(osize, osize);

                idata.setRandom();
                kdata.setRandom();
                odata.setRandom();

                idata /= isize;
                kdata /= ksize;
                odata /= osize;

                const scalar_t corrcpu_egb = test_cpu(ncv::corr2d_egb<matrix_t>, odata, kdata, idata);
                const scalar_t corrcpu_egr = test_cpu(ncv::corr2d_egr<matrix_t>, odata, kdata, idata);
                const scalar_t corrcpu_cpp = test_cpu(ncv::corr2d_cpp<matrix_t>, odata, kdata, idata);
                const scalar_t corrcpu_mdk = test_cpu(ncv::corr2d_mdk<matrix_t>, odata, kdata, idata);
                const scalar_t corrcpu_mdo = test_cpu(ncv::corr2d_mdo<matrix_t>, odata, kdata, idata);
                const scalar_t corrcpu_dyn = test_cpu(ncv::corr2d_dyn<matrix_t>, odata, kdata, idata);
        #if defined(NANOCV_WITH_OPENCL)
                const scalar_t corrgpu   = test_gpu("corr_kernel", "gpu", odata, kdata, idata);
        #elif NANOCV_WITH_CUDA
                const scalar_t corrgpu   = test_gpu(cuda::corr2d<scalar_t>, odata, kdata, idata);
        #endif

                const scalar_t epsilon = math::epsilon1<scalar_t>();

                BOOST_CHECK_LE(math::abs(corrcpu_egb - corrcpu_egb), epsilon);
                BOOST_CHECK_LE(math::abs(corrcpu_egr - corrcpu_egb), epsilon);
                BOOST_CHECK_LE(math::abs(corrcpu_cpp - corrcpu_egb), epsilon);
                BOOST_CHECK_LE(math::abs(corrcpu_mdk - corrcpu_egb), epsilon);
                BOOST_CHECK_LE(math::abs(corrcpu_mdo - corrcpu_egb), epsilon);
                BOOST_CHECK_LE(math::abs(corrcpu_dyn - corrcpu_egb), epsilon);
        #if defined(NANOCV_WITH_OPENCL) || defined(NANOCV_WITH_CUDA)
                BOOST_CHECK_LE(math::abs(corrgpu     - corrcpu_egb), epsilon);
        #endif
        }
}

BOOST_AUTO_TEST_CASE(test_corr2d)
{
        using namespace ncv;

#ifdef NANOCV_WITH_OPENCL
        if (!ocl::get_manager().valid())
        {
                BOOST_CHECK_EQUAL(true, false);
                exit(EXIT_FAILURE);
        }
#endif

#ifdef NANOCV_WITH_CUDA
        cuda::print_info();
#endif

        const int min_isize = 24;
        const int max_isize = 48;
        const int min_ksize = 5;
        const int n_tests = 64;

#ifdef NANOCV_WITH_OPENCL
        try
#endif
        {
                for (int isize = min_isize; isize <= max_isize; isize += 4)
                {
                        for (int ksize = min_ksize; ksize <= isize - min_ksize; ksize += 2)
                        {
                                for (int t = 0; t < n_tests; t ++)
                                {
                                        test::test_corr2d(isize, ksize);
                                }
                        }
                }
        }

#ifdef NANOCV_WITH_OPENCL
        catch (cl::Error& e)
        {
                log_error() << "OpenCL fatal error: <" << e.what() << "> (" << ocl::error_string(e.err()) << ")!";
                BOOST_CHECK_EQUAL(true, false);
        }
#endif
}

