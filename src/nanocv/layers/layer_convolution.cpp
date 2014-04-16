#include "layer_convolution.h"
#include "text.h"
#include "common/logger.h"
#include "common/math.hpp"
#include "common/random.hpp"

namespace ncv
{
        /////////////////////////////////////////////////////////////////////////////////////////

        template
        <
                typename tscalar,
                typename tsize
        >
        static void _forward(
                const tscalar* idata, tsize idims,
                const tscalar* kdata, tsize krows, tsize kcols,
                tscalar* odata, tsize odims, tsize orows, tsize ocols)
        {
                const tsize irows = orows + krows - 1;
                const tsize icols = ocols + kcols - 1;
                const tsize isize = irows * icols;

                const tsize osize = orows * ocols;
                const tsize ksize = krows * kcols;

                std::fill(odata, odata + odims * osize, tscalar(0));

                // output
                for (tsize o = 0; o < odims; o ++)
                {
                        tscalar* podata = odata + o * osize;

                        for (tsize i = 0; i < idims; i ++)
                        {
                                const tscalar* pidata = idata + i * isize;
                                const tscalar* pkdata = kdata + (o * idims + i) * ksize;

                                for (tsize r = 0; r < orows; r ++)
                                {
                                        for (tsize c = 0; c < ocols; c ++)
                                        {
                                                tscalar sum = 0;
                                                for (tsize kr = 0; kr < krows; kr ++)
                                                {
                                                        for (tsize kc = 0; kc < kcols; kc ++)
                                                        {
                                                                const tscalar iv = pidata[(r + kr) * icols + (c + kc)];
                                                                const tscalar kv = pkdata[kr * kcols + kc];

                                                                sum += iv * kv;
                                                        }
                                                }

                                                podata[r * ocols + c] += sum;
                                        }
                                }
                        }
                }
        }

        /////////////////////////////////////////////////////////////////////////////////////////

        template
        <
                typename tscalar,
                typename tsize
        >
        static void _backward(
                const tscalar* idata, tscalar* gidata, tsize idims,
                const tscalar* kdata, tscalar* gkdata, tsize krows, tsize kcols,
                const tscalar* odata, tsize odims, tsize orows, tsize ocols)
        {
                const tsize irows = orows + krows - 1;
                const tsize icols = ocols + kcols - 1;
                const tsize isize = irows * icols;

                const tsize osize = orows * ocols;
                const tsize ksize = krows * kcols;

                std::fill(gidata, gidata + idims * isize, tscalar(0));
                std::fill(gkdata, gkdata + odims * idims * ksize, tscalar(0));

                for (tsize o = 0; o < odims; o ++)
                {
                        const tscalar* podata = odata + o * osize;

                        for (tsize i = 0; i < idims; i ++)
                        {
                                const tscalar* pidata = idata + i * isize;                                
                                const tscalar* pkdata = kdata + (o * idims + i) * ksize;

                                tscalar* pgidata = gidata + i * isize;
                                tscalar* pgkdata = gkdata + (o * idims + i) * ksize;

                                for (tsize r = 0; r < orows; r ++)
                                {
                                        for (tsize c = 0; c < ocols; c ++)
                                        {
                                                for (tsize kr = 0; kr < krows; kr ++)
                                                {
                                                        for (tsize kc = 0; kc < kcols; kc ++)
                                                        {
                                                                const tscalar iv = pidata[(r + kr) * icols + (c + kc)];
                                                                const tscalar ov = podata[r * ocols + c];
                                                                const tscalar kv = pkdata[kr * kcols + kc];

                                                                pgidata[(r + kr) * icols + (c + kc)] += ov * kv;
                                                                pgkdata[kr * kcols + kc] += ov * iv;
                                                        }
                                                }
                                        }
                                }
                        }
                }
        }

        /////////////////////////////////////////////////////////////////////////////////////////

#if NANOCV_HAVE_OPENCL
        static const string_t ocl_conv_source = R"xxx(

        #pragma OPENCL EXTENSION cl_amd_fp64 : enable
        #pragma OPENCL EXTENSION cl_khr_fp64 : enable

        __kernel void conv_forward(
                __global const double* idata, int idims,
                __global const double* kdata, int krows, int kcols,
                __constant const double* wdata,
                __global double* odata)
        {
                const int odims = get_global_size(0);
                const int orows = get_global_size(1);
                const int ocols = get_global_size(2);
                const int osize = orows * ocols;

                const int icols = ocols + kcols - 1;
                const int irows = orows + krows - 1;
                const int isize = irows * icols;

                const int ksize = krows * kcols;

                const int o = get_global_id(0);
                const int r = get_global_id(1);
                const int c = get_global_id(2);

                double sum_conv = 0;
                for (int i = 0; i < idims; i ++)
                {
                        const double w = wdata[o * idims + i];

                        double sum = 0;
                        for (int kr = 0; kr < krows; kr ++)
                        {
                                for (int kc = 0; kc < kcols; kc ++)
                                {
                                        const double iv = idata[i * isize + (r + kr) * icols + (c + kc)];
                                        const double kv = kdata[o * ksize + kr * kcols + kc];

                                        sum += iv * kv;
                                }
                        }

                        sum_conv += w * sum;
                }

                odata[o * osize + r * ocols + c] = sum_conv;
        }

        __kernel void conv_ibackward(
                __global const double* odata, int odims,
                __global const double* kdata, int krows, int kcols,
                __constant const double* wdata,
                __global double* gidata)
        {
                const int idims = get_global_size(0);
                const int irows = get_global_size(1);
                const int icols = get_global_size(2);
                const int isize = irows * icols;

                const int ocols = icols - kcols + 1;
                const int orows = irows - krows + 1;
                const int osize = orows * ocols;

                const int ksize = krows * kcols;

                const int i = get_global_id(0);
                const int r = get_global_id(1);
                const int c = get_global_id(2);

                const int krmin = max(0,     r - orows + 1);
                const int krmax = min(krows, r + 1);

                const int kcmin = max(0,     c - ocols + 1);
                const int kcmax = min(kcols, c + 1);

                double sum_conv = 0;
                for (int o = 0; o < odims; o ++)
                {
                        const double w = wdata[o * idims + i];
                        __global const double* podata = odata + o * osize;
                        __global const double* pkdata = kdata + o * ksize;

                        double sum = 0;
                        for (int kr = krmin; kr < krmax; kr ++)
                        {
                                for (int kc = kcmin; kc < kcmax; kc ++)
                                {
                                        const double ov = podata[(r - kr) * ocols + (c - kc)];
                                        const double kv = pkdata[kr * kcols + kc];

                                        sum += ov * kv;
                                }
                        }

                        sum_conv += w * sum;
                }

                gidata[i * isize + r * icols + c] = sum_conv;
        }

        __kernel void conv_kbackward(
                __global const double* odata, int orows, int ocols,
                __global const double* idata, int idims,
                __constant const double* wdata,
                __global double* gkdata)
        {
                const int odims = get_global_size(0);
                const int krows = get_global_size(1);
                const int kcols = get_global_size(2);
                const int ksize = krows * kcols;

                const int irows = orows + krows - 1;
                const int icols = ocols + kcols - 1;
                const int isize = irows * icols;

                const int osize = orows * ocols;

                const int o = get_global_id(0);
                const int kr = get_global_id(1);
                const int kc = get_global_id(2);

                double sum_conv = 0;
                for (int i = 0; i < idims; i ++)
                {
                        const double w = wdata[o * idims + i];
                        __global const double* podata = odata + o * osize;
                        __global const double* pidata = idata + i * isize;

                        double sum = 0;
                        for (int r = 0; r < orows; r ++)
                        {
                                for (int c = 0; c < ocols; c ++)
                                {
                                        const double ov = podata[r * ocols + c];
                                        const double iv = pidata[(r + kr) * icols + (c + kc)];

                                        sum += ov * iv;
                                }
                        }

                        sum_conv += w * sum;
                }

                gkdata[o * ksize + kr * kcols + kc] = sum_conv;
        }

        __kernel void conv_wbackward(
                __global const double* odata, int orows, int ocols,
                __global const double* idata,
                __global const double* kdata, int krows, int kcols,
                __global double* gwdata)
        {
                const int odims = get_global_size(0);
                const int idims = get_global_size(1);

                const int irows = orows + krows - 1;
                const int icols = ocols + kcols - 1;
                const int isize = irows * icols;

                const int osize = orows * ocols;
                const int ksize = krows * kcols;

                const int o = get_global_id(0);
                const int i = get_global_id(1);

                __global const double* podata = odata + o * osize;
                __global const double* pidata = idata + i * isize;
                __global const double* pkdata = kdata + o * ksize;

                double sum_conv = 0;
                for (int r = 0; r < orows; r ++)
                {
                        for (int c = 0; c < ocols; c ++)
                        {
                                const double ov = podata[r * ocols + c];

                                double sum = 0;
                                for (int kr = 0; kr < krows; kr ++)
                                {
                                        for (int kc = 0; kc < kcols; kc ++)
                                        {
                                                const double iv = pidata[(r + kr) * icols + (c + kc)];
                                                const double kv = pkdata[kr * kcols + kc];

                                                sum += iv * kv;
                                        }
                                }

                                sum_conv += ov * sum;
                        }
                }

                gwdata[o * idims + i] = sum_conv;
        }

        )xxx";
#endif

        /////////////////////////////////////////////////////////////////////////////////////////

        conv_layer_t::conv_layer_t(const string_t& parameters)
                :       layer_t(parameters, "convolution layer, parameters: dims=16[1,256],rows=8[1,32],cols=8[1,32]")
        {
        }

        /////////////////////////////////////////////////////////////////////////////////////////

        size_t conv_layer_t::resize(const tensor_t& tensor)
        {
                const size_t idims = tensor.dims();
                const size_t irows = tensor.rows();
                const size_t icols = tensor.cols();

                const size_t odims = math::clamp(text::from_params<size_t>(parameters(), "dims", 16), 1, 256);
                const size_t krows = math::clamp(text::from_params<size_t>(parameters(), "rows", 8), 1, 32);
                const size_t kcols = math::clamp(text::from_params<size_t>(parameters(), "cols", 8), 1, 32);

                if (irows < krows || icols < kcols)
                {
                        const string_t message =
                                "invalid size (" + text::to_string(idims) + "x" + text::to_string(irows) +
                                 "x" + text::to_string(icols) + ") -> (" + text::to_string(odims) + "x" +
                                 text::to_string(krows) + "x" + text::to_string(kcols) + ")";

                        log_error() << "convolution layer: " << message;
                        throw std::runtime_error("convolution layer: " + message);
                }

                const size_t orows = irows - krows + 1;
                const size_t ocols = icols - kcols + 1;

                // resize buffers
                m_idata.resize(idims, irows, icols);
                m_odata.resize(odims, orows, ocols);

                m_kdata.resize(odims * idims, krows, kcols);

                m_gkdata.resize(odims * idims, krows, kcols);
                m_gidata.resize(idims, irows, icols);

#if NANOCV_HAVE_OPENCL
                // create opencl objects (if available)
                ocl::manager_t& theocl = ocl::manager_t::instance();
                if (theocl.valid())
                {
                        // kernels
                        m_ocl_queue = theocl.make_command_queue();
                        m_ocl_program = theocl.make_program_from_text(ocl_conv_source);
                        m_ocl_fkernel = theocl.make_kernel(m_ocl_program, "conv_forward");
                        m_ocl_bikernel = theocl.make_kernel(m_ocl_program, "conv_ibackward");
                        m_ocl_bkkernel = theocl.make_kernel(m_ocl_program, "conv_kbackward");
                        m_ocl_bwkernel = theocl.make_kernel(m_ocl_program, "conv_wbackward");

                        // forward buffers
                        m_ocl_idata = theocl.make_buffer(m_idata.size() * sizeof(scalar_t), CL_MEM_READ_ONLY);
                        m_ocl_kdata = theocl.make_buffer(m_kdata.size() * sizeof(scalar_t), CL_MEM_READ_ONLY);
                        m_ocl_wdata = theocl.make_buffer(m_wdata.size() * sizeof(scalar_t), CL_MEM_READ_ONLY);
                        m_ocl_odata = theocl.make_buffer(m_odata.size() * sizeof(scalar_t), CL_MEM_READ_WRITE);

                        // backward buffers
                        m_ocl_gidata = theocl.make_buffer(m_gidata.size() * sizeof(scalar_t), CL_MEM_WRITE_ONLY);
                        m_ocl_gkdata = theocl.make_buffer(m_gkdata.size() * sizeof(scalar_t), CL_MEM_WRITE_ONLY);
                        m_ocl_gwdata = theocl.make_buffer(m_gwdata.size() * sizeof(scalar_t), CL_MEM_WRITE_ONLY);

                        const int idims_ = static_cast<int>(idims);

                        const int krows_ = static_cast<int>(krows);
                        const int kcols_ = static_cast<int>(kcols);

                        const int odims_ = static_cast<int>(odims);
                        const int orows_ = static_cast<int>(orows);
                        const int ocols_ = static_cast<int>(ocols);

                        // setup forward kernel
                        m_ocl_fkernel.setArg(0, m_ocl_idata);
                        m_ocl_fkernel.setArg(1, sizeof(int), (void*)&idims_);
                        m_ocl_fkernel.setArg(2, m_ocl_kdata);
                        m_ocl_fkernel.setArg(3, sizeof(int), (void*)&krows_);
                        m_ocl_fkernel.setArg(4, sizeof(int), (void*)&kcols_);
                        m_ocl_fkernel.setArg(5, m_ocl_wdata);
                        m_ocl_fkernel.setArg(6, m_ocl_odata);

                        // setup backward kernels
                        m_ocl_bikernel.setArg(0, m_ocl_odata);
                        m_ocl_bikernel.setArg(1, sizeof(int), (void*)&odims_);
                        m_ocl_bikernel.setArg(2, m_ocl_kdata);
                        m_ocl_bikernel.setArg(3, sizeof(int), (void*)&krows_);
                        m_ocl_bikernel.setArg(4, sizeof(int), (void*)&kcols_);
                        m_ocl_bikernel.setArg(5, m_ocl_wdata);
                        m_ocl_bikernel.setArg(6, m_ocl_gidata);

                        m_ocl_bkkernel.setArg(0, m_ocl_odata);
                        m_ocl_bkkernel.setArg(1, sizeof(int), (void*)&orows_);
                        m_ocl_bkkernel.setArg(2, sizeof(int), (void*)&ocols_);
                        m_ocl_bkkernel.setArg(3, m_ocl_idata);
                        m_ocl_bkkernel.setArg(4, sizeof(int), (void*)&idims_);
                        m_ocl_bkkernel.setArg(5, m_ocl_wdata);
                        m_ocl_bkkernel.setArg(6, m_ocl_gkdata);

                        m_ocl_bwkernel.setArg(0, m_ocl_odata);
                        m_ocl_bwkernel.setArg(1, sizeof(int), (void*)&orows_);
                        m_ocl_bwkernel.setArg(2, sizeof(int), (void*)&ocols_);
                        m_ocl_bwkernel.setArg(3, m_ocl_idata);
                        m_ocl_bwkernel.setArg(4, m_ocl_kdata);
                        m_ocl_bwkernel.setArg(5, sizeof(int), (void*)&krows_);
                        m_ocl_bwkernel.setArg(6, sizeof(int), (void*)&kcols_);
                        m_ocl_bwkernel.setArg(7, m_ocl_gwdata);
                }
#endif

                return n_parameters();
        }

        /////////////////////////////////////////////////////////////////////////////////////////

        void conv_layer_t::zero_params()
        {
                m_kdata.zero();

                params_changed();
        }

        /////////////////////////////////////////////////////////////////////////////////////////

        void conv_layer_t::random_params(scalar_t min, scalar_t max)
        {
                m_kdata.random(random_t<scalar_t>(min, max));

                params_changed();
        }

        /////////////////////////////////////////////////////////////////////////////////////////

        ovectorizer_t& conv_layer_t::save_params(ovectorizer_t& s) const
        {
                return s << m_kdata;
        }

        /////////////////////////////////////////////////////////////////////////////////////////

        ovectorizer_t& conv_layer_t::save_grad(ovectorizer_t& s) const
        {
                return s << m_gkdata;
        }

        /////////////////////////////////////////////////////////////////////////////////////////

        ivectorizer_t& conv_layer_t::load_params(ivectorizer_t& s)
        {
                s >> m_kdata;

                params_changed();

                return s;
        }

        /////////////////////////////////////////////////////////////////////////////////////////

        void conv_layer_t::params_changed() const
        {
#if NANOCV_HAVE_OPENCL
                // send parameters to OpenCL device (if available)
                ocl::manager_t& theocl = ocl::manager_t::instance();
                if (theocl.valid())
                {
                        m_ocl_queue.enqueueWriteBuffer(m_ocl_kdata, CL_TRUE, 0, m_kdata.size() * sizeof(scalar_t), m_kdata.data());
                        m_ocl_queue.enqueueWriteBuffer(m_ocl_wdata, CL_TRUE, 0, m_wdata.size() * sizeof(scalar_t), m_wdata.data());
                }
#endif
        }

        /////////////////////////////////////////////////////////////////////////////////////////

        const tensor_t& conv_layer_t::forward(const tensor_t& input)
        {
                assert(idims() == input.dims());
                assert(irows() == input.rows());
                assert(icols() == input.cols());

                m_idata.copy_from(input);

#if NANOCV_HAVE_OPENCL
                // OpenCL version
                ocl::manager_t& theocl = ocl::manager_t::instance();
                if (theocl.valid())
                {
                        m_ocl_queue.enqueueWriteBuffer(m_ocl_idata, CL_TRUE, 0, m_idata.size() * sizeof(scalar_t), m_idata.data());

                        m_ocl_queue.enqueueNDRangeKernel(m_ocl_fkernel, cl::NullRange,
                                cl::NDRange(odims(), orows(), ocols()),
                                cl::NDRange(1, orows(), ocols()));
                        m_ocl_queue.finish();

                        m_ocl_queue.enqueueReadBuffer(m_ocl_odata, CL_TRUE, 0, m_odata.size() * sizeof(scalar_t), m_odata.data());
                }

                // CPU version
                else
#endif
                {
                        _forward(m_idata.data(), idims(),
                                 m_kdata.data(), krows(), kcols(),
                                 m_odata.data(), odims(), orows(), ocols());
                }

                return m_odata;
        }        
        
	/////////////////////////////////////////////////////////////////////////////////////////

        const tensor_t& conv_layer_t::backward(const tensor_t& gradient)
        {
                assert(odims() == gradient.dims());
                assert(orows() == gradient.rows());
                assert(ocols() == gradient.cols());

		m_odata.copy_from(gradient);

#if NANOCV_HAVE_OPENCL
                // OpenCL version
                ocl::manager_t& theocl = ocl::manager_t::instance();
                if (theocl.valid())
                {
                        m_ocl_queue.enqueueWriteBuffer(m_ocl_odata, CL_TRUE, 0, m_odata.size() * sizeof(scalar_t), m_odata.data());
                        m_ocl_queue.enqueueWriteBuffer(m_ocl_idata, CL_TRUE, 0, m_idata.size() * sizeof(scalar_t), m_idata.data());

                        m_ocl_queue.enqueueNDRangeKernel(m_ocl_bikernel, cl::NullRange,
                                cl::NDRange(idims(), irows(), icols()),
                                cl::NDRange(1, irows(), icols()));

                        m_ocl_queue.enqueueNDRangeKernel(m_ocl_bkkernel, cl::NullRange,
                                cl::NDRange(odims(), krows(), kcols()),
                                cl::NDRange(1, krows(), kcols()));

                        m_ocl_queue.enqueueNDRangeKernel(m_ocl_bwkernel, cl::NullRange,
                                cl::NDRange(odims(), idims()),
                                cl::NDRange(odims(), idims()));
                        m_ocl_queue.finish();

                        m_ocl_queue.enqueueReadBuffer(m_ocl_gidata, CL_TRUE, 0, m_gidata.size() * sizeof(scalar_t), m_gidata.data());
                        m_ocl_queue.enqueueReadBuffer(m_ocl_gkdata, CL_TRUE, 0, m_gkdata.size() * sizeof(scalar_t), m_gkdata.data());
                        m_ocl_queue.enqueueReadBuffer(m_ocl_gwdata, CL_TRUE, 0, m_gwdata.size() * sizeof(scalar_t), m_gwdata.data());
                }

                // CPU version
                else
#endif
                {
                        _backward(m_idata.data(), m_gidata.data(), idims(),
                                  m_kdata.data(), m_gkdata.data(), krows(), kcols(),
                                  m_odata.data(), odims(), orows(), ocols());
                }

                return m_gidata;
        }

        /////////////////////////////////////////////////////////////////////////////////////////
}


