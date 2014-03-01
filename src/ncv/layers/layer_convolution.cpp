#include "layer_convolution.h"
#include "text.h"
#include "common/logger.h"
#include "common/math.hpp"
#include "common/random.hpp"
#include "opencl/opencl.h"

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
                const tscalar* wdata,
                tscalar* odata, tsize odims, tsize orows, tsize ocols)
        {
                const tsize irows = orows + krows - 1;
                const tsize icols = ocols + kcols - 1;
                const tsize isize = irows * icols;

                const tsize osize = orows * ocols;
                const tsize ksize = krows * kcols;

                for (tsize o = 0; o < odims; o ++)
                {
                        tscalar* podata = odata + o * osize;

                        for (tsize r = 0; r < orows; r ++)
                        {
                                for (tsize c = 0; c < ocols; c ++)
                                {
                                        podata[r * ocols + c] = 0;
                                }
                        }

                        for (tsize i = 0; i < idims; i ++)
                        {
                                const tscalar* pidata = idata + i * isize;
                                const tscalar w = wdata[o * idims + i];

                                for (tsize r = 0; r < orows; r ++)
                                {
                                        for (tsize c = 0; c < ocols; c ++)
                                        {
                                                const tscalar* pkdata = kdata + o * ksize;

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

                                                podata[r * ocols + c] += w * sum;
                                        }
                                }
                        }
                }
        }

        /////////////////////////////////////////////////////////////////////////////////////////

        static const string_t conv_forward_source = R"xxx(

        #pragma OPENCL EXTENSION cl_amd_fp64 : enable
        #pragma OPENCL EXTENSION cl_khr_fp64 : enable

        __kernel void conv_forward(
                __global const double* idata, int idims,
                __constant const double* kdata, int krows, int kcols,
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

                __global double* podata = odata + o * osize;
                __constant const double* pkdata = kdata + o * ksize;

                double sum_conv = 0;
                for (int i = 0; i < idims; i ++)
                {
                        __global const double* pidata = idata + i * isize;
                        const double w = wdata[o * idims + i];

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

                        sum_conv += w * sum;
                }

                podata[r * ocols + c] = sum_conv;
        }

        )xxx";

        /////////////////////////////////////////////////////////////////////////////////////////

        template
        <
                typename tscalar,
                typename tsize
        >
        static void _backward(
                const tscalar* idata, tscalar* gidata, tsize idims,
                const tscalar* kdata, tscalar* gkdata, tsize krows, tsize kcols,
                const tscalar* wdata, tscalar* gwdata,
                const tscalar* odata, tsize odims, tsize orows, tsize ocols)
        {
                const tsize irows = orows + krows - 1;
                const tsize icols = ocols + kcols - 1;
                const tsize isize = irows * icols;

                const tsize osize = orows * ocols;
                const tsize ksize = krows * kcols;

                for (tsize o = 0; o < odims; o ++)
                {
                        for (tsize i = 0; i < idims; i ++)
                        {
                                gwdata[o * idims + i] = 0;
                        }

                        tscalar* pgkdata = gkdata + o * ksize;
                        for (tsize kr = 0; kr < krows; kr ++)
                        {
                                for (tsize kc = 0; kc < kcols; kc ++)
                                {
                                        pgkdata[kr * kcols + kc] = 0;
                                }
                        }
                }

                for (tsize i = 0; i < idims; i ++)
                {
                        tscalar* pgidata = gidata + i * isize;
                        for (tsize ir = 0; ir < irows; ir ++)
                        {
                                for (tsize ic = 0; ic < icols; ic ++)
                                {
                                        pgidata[ir * icols + ic] = 0;
                                }
                        }
                }

                for (tsize o = 0; o < odims; o ++)
                {
                        const tscalar* podata = odata + o * osize;
                        const tscalar* pkdata = kdata + o * ksize;

                        tscalar* pgkdata = gkdata + o * ksize;

                        for (tsize i = 0; i < idims; i ++)
                        {
                                const tscalar* pidata = idata + i * isize;
                                const tscalar w = wdata[o * idims + i];

                                tscalar* pgidata = gidata + i * isize;
                                tscalar& gw = gwdata[o * idims + i];

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

                                                                pgidata[(r + kr) * icols + (c + kc)] += ov * kv * w;
                                                                pgkdata[kr * kcols + kc] += ov * iv * w;
                                                                gw += ov * iv * kv;
                                                        }
                                                }
                                        }
                                }
                        }
                }
        }

        /////////////////////////////////////////////////////////////////////////////////////////

        conv_layer_t::conv_layer_t(const string_t& params)
                :       m_params(params),
                        m_ocl_idata_id(0),
                        m_ocl_odata_id(0),
                        m_ocl_kdata_id(0),
                        m_ocl_wdata_id(0),
                        m_ocl_gkdata_id(0),
                        m_ocl_gwdata_id(0),
                        m_ocl_gidata_id(0),
                        m_ocl_fkernel_id(0),
                        m_ocl_bkernel_id(0)
        {
        }

        /////////////////////////////////////////////////////////////////////////////////////////

        size_t conv_layer_t::resize(const tensor_t& tensor)
        {
                const size_t idims = tensor.dims();
                const size_t irows = tensor.rows();
                const size_t icols = tensor.cols();

                const size_t odims = math::clamp(text::from_params<size_t>(m_params, "dims", 16), 1, 256);
                const size_t krows = math::clamp(text::from_params<size_t>(m_params, "rows", 8), 1, 32);
                const size_t kcols = math::clamp(text::from_params<size_t>(m_params, "cols", 8), 1, 32);

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

                m_kdata.resize(odims, krows, kcols);
                m_wdata.resize(1, odims, idims);

                m_gkdata.resize(odims, krows, kcols);
                m_gwdata.resize(1, odims, idims);
                m_gidata.resize(idims, irows, icols);

                // create opencl objects (if available)
                ocl::manager_t& theocl = ocl::manager_t::instance();
                if (theocl.valid())
                {
                        const size_t ocl_program_forward_id = theocl.make_program_from_text(conv_forward_source);
                        m_ocl_fkernel_id = theocl.make_kernel(ocl_program_forward_id, "conv_forward");

                        // forward buffers
                        m_ocl_idata_id = theocl.make_buffer(m_idata.size() * sizeof(scalar_t), CL_MEM_READ_ONLY);
                        m_ocl_kdata_id = theocl.make_buffer(m_kdata.size() * sizeof(scalar_t), CL_MEM_READ_ONLY);
                        m_ocl_wdata_id = theocl.make_buffer(m_wdata.size() * sizeof(scalar_t), CL_MEM_READ_ONLY);
                        m_ocl_odata_id = theocl.make_buffer(m_odata.size() * sizeof(scalar_t), CL_MEM_WRITE_ONLY);

                        // setup forward kernel
                        theocl.set_kernel_buffer(m_ocl_fkernel_id, 0, m_ocl_idata_id);
                        theocl.set_kernel_integer(m_ocl_fkernel_id, 1, idims);
                        theocl.set_kernel_buffer(m_ocl_fkernel_id, 2, m_ocl_kdata_id);
                        theocl.set_kernel_integer(m_ocl_fkernel_id, 3, krows);
                        theocl.set_kernel_integer(m_ocl_fkernel_id, 4, kcols);
                        theocl.set_kernel_buffer(m_ocl_fkernel_id, 5, m_ocl_wdata_id);
                        theocl.set_kernel_buffer(m_ocl_fkernel_id, 6, m_ocl_odata_id);
                        theocl.finish();
                }

                return m_kdata.size() + m_wdata.size();
        }

        /////////////////////////////////////////////////////////////////////////////////////////

        void conv_layer_t::zero_params()
        {
                m_kdata.zero();
                m_wdata.zero();

                params_changed();
        }

        /////////////////////////////////////////////////////////////////////////////////////////

        void conv_layer_t::random_params(scalar_t min, scalar_t max)
        {
                m_kdata.random(random_t<scalar_t>(min, max));
                m_wdata.random(random_t<scalar_t>(min, max));

                params_changed();
        }

        /////////////////////////////////////////////////////////////////////////////////////////

        ovectorizer_t& conv_layer_t::save_params(ovectorizer_t& s) const
        {
                return s << m_kdata << m_wdata;
        }

        /////////////////////////////////////////////////////////////////////////////////////////

        ovectorizer_t& conv_layer_t::save_grad(ovectorizer_t& s) const
        {
                return s << m_gkdata << m_gwdata;
        }

        /////////////////////////////////////////////////////////////////////////////////////////

        ivectorizer_t& conv_layer_t::load_params(ivectorizer_t& s)
        {
                s >> m_kdata >> m_wdata;

                params_changed();

                return s;
        }        

        /////////////////////////////////////////////////////////////////////////////////////////

        void conv_layer_t::params_changed() const
        {
                // send parameters to OpenCL device (if available)
                ocl::manager_t& theocl = ocl::manager_t::instance();
                if (theocl.valid())
                {
                        cl::Event evk, evw;
                        theocl.write_buffer(m_ocl_kdata_id, m_kdata.size() * sizeof(scalar_t), m_kdata.data(), &evk);
                        theocl.write_buffer(m_ocl_wdata_id, m_wdata.size() * sizeof(scalar_t), m_wdata.data(), &evw);
                        evk.wait();
                        evw.wait();
                }
        }

        /////////////////////////////////////////////////////////////////////////////////////////

        const tensor_t& conv_layer_t::forward(const tensor_t& input)
        {
                assert(idims() == input.dims());
                assert(irows() == input.rows());
                assert(icols() == input.cols());

                m_idata.copy_from(input);

                // OpenCL version
                ocl::manager_t& theocl = ocl::manager_t::instance();
                if (theocl.valid())
                {
                        cl::Event evi;
                        theocl.write_buffer(m_ocl_idata_id, m_idata.size() * sizeof(scalar_t), m_idata.data(), &evi);
                        evi.wait();

                        cl::Event evrun;
                        theocl.run_kernel(m_ocl_fkernel_id,
                                          cl::NDRange(odims(), orows(), ocols()),
                                          cl::NDRange(1, orows(), ocols()), &evrun);
                        evrun.wait();

                        cl::Event evo;
                        theocl.read_buffer(m_ocl_odata_id, m_odata.size() * sizeof(scalar_t), m_odata.data(), &evo);
                        evo.wait();
                }

                // CPU version
                else
                {
                        _forward(m_idata.data(), idims(),
                                 m_kdata.data(), krows(), kcols(),
                                 m_wdata.data(),
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

		_backward(m_idata.data(), m_gidata.data(), idims(),
			  m_kdata.data(), m_gkdata.data(), krows(), kcols(),
			  m_wdata.data(), m_gwdata.data(),
			  m_odata.data(), odims(), orows(), ocols());

                return m_gidata;
        }

        /////////////////////////////////////////////////////////////////////////////////////////
}


