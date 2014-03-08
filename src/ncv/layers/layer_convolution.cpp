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
                const tscalar* wdata,
                tscalar* odata, tsize odims, tsize orows, tsize ocols)
        {
                const tsize irows = orows + krows - 1;
                const tsize icols = ocols + kcols - 1;
                const tsize isize = irows * icols;

                const tsize osize = orows * ocols;
                const tsize ksize = krows * kcols;

                std::fill(odata, odata + odims * osize, tscalar(0));

                for (tsize o = 0; o < odims; o ++)
                {
                        tscalar* podata = odata + o * osize;

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

                std::fill(gidata, gidata + idims * isize, tscalar(0));
                std::fill(gkdata, gkdata + odims * ksize, tscalar(0));
                std::fill(gwdata, gwdata + odims * idims, tscalar(0));

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

                m_kdata.resize(odims, krows, kcols);
                m_wdata.resize(1, odims, idims);

                m_gkdata.resize(odims, krows, kcols);
                m_gwdata.resize(1, odims, idims);
                m_gidata.resize(idims, irows, icols);

                return m_kdata.size() + m_wdata.size();
        }

        /////////////////////////////////////////////////////////////////////////////////////////

        void conv_layer_t::zero_params()
        {
                m_kdata.zero();
                m_wdata.zero();
        }

        /////////////////////////////////////////////////////////////////////////////////////////

        void conv_layer_t::random_params(scalar_t min, scalar_t max)
        {
                m_kdata.random(random_t<scalar_t>(min, max));
                m_wdata.random(random_t<scalar_t>(min, max));
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
                return s >> m_kdata >> m_wdata;
        }

        /////////////////////////////////////////////////////////////////////////////////////////

        const tensor_t& conv_layer_t::forward(const tensor_t& input)
        {
                assert(idims() == input.dims());
                assert(irows() == input.rows());
                assert(icols() == input.cols());

                m_idata.copy_from(input);

                _forward(m_idata.data(), idims(),
                         m_kdata.data(), krows(), kcols(),
                         m_wdata.data(),
                         m_odata.data(), odims(), orows(), ocols());

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


