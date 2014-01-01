#include "layer_convolution.h"
#include "text.h"
#include "vectorizer.h"
#include "util/logger.h"
#include "util/math.hpp"
#include "util/dot.hpp"
#include "util/mad.hpp"
#include "util/convolution.hpp"

namespace ncv
{
        /////////////////////////////////////////////////////////////////////////////////////////

        conv_layer_t::conv_layer_t(const string_t& params)
                :       m_params(params)
        {
        }

        /////////////////////////////////////////////////////////////////////////////////////////

        size_t conv_layer_t::resize(size_t idims, size_t irows, size_t icols)
        {
                const size_t odims = math::clamp(text::from_params<size_t>(m_params, "count", 16), 1, 256);
                const size_t crows = math::clamp(text::from_params<size_t>(m_params, "rows", 8), 1, 32);
                const size_t ccols = math::clamp(text::from_params<size_t>(m_params, "cols", 8), 1, 32);

                if (irows < crows || icols < ccols)
                {
                        const string_t message =
                                "invalid size (" + text::to_string(idims) + "x" + text::to_string(irows) +
                                 "x" + text::to_string(icols) + ") -> (" + text::to_string(odims) + "x" +
                                 text::to_string(crows) + "x" + text::to_string(ccols) + ")";

                        log_error() << "convolution layer: " << message;
                        throw std::runtime_error("convolution layer: " + message);
                }

                const size_t orows = irows - crows + 1;
                const size_t ocols = icols - ccols + 1;

                m_idata.resize(idims, irows, icols);
                m_odata.resize(odims, orows, ocols);
                m_xdata.resize(odims, idims, orows, ocols);

                m_kdata.resize(odims, crows, ccols);
                m_wdata.resize(odims, idims, 1);
                m_bdata.resize(odims, 1, 1);

                m_gkdata.resize(odims, crows, ccols);
                m_gwdata.resize(odims, idims, 1);
                m_gbdata.resize(odims, 1, 1);

                return m_kdata.size() + m_wdata.size() + m_bdata.size();
        }

        /////////////////////////////////////////////////////////////////////////////////////////

        void conv_layer_t::zero_params()
        {
                m_kdata.zero();
                m_wdata.zero();
                m_bdata.zero();
        }

        /////////////////////////////////////////////////////////////////////////////////////////

        void conv_layer_t::random_params(scalar_t min, scalar_t max)
        {
                m_kdata.random(min, max);
                m_wdata.random(min, max);
                m_bdata.random(min, max);
        }

        /////////////////////////////////////////////////////////////////////////////////////////

        ovectorizer_t& conv_layer_t::save_params(ovectorizer_t& s) const
        {
                return s << m_kdata << m_bdata << m_wdata;
        }

        /////////////////////////////////////////////////////////////////////////////////////////

        ovectorizer_t& conv_layer_t::save_grad(ovectorizer_t& s) const
        {
                return s << m_gkdata << m_gbdata << m_gwdata;
        }

        /////////////////////////////////////////////////////////////////////////////////////////

        ivectorizer_t& conv_layer_t::load_params(ivectorizer_t& s)
        {
                return s >> m_kdata >> m_bdata >> m_wdata;
        }

        /////////////////////////////////////////////////////////////////////////////////////////

        const tensor3d_t& conv_layer_t::forward(const tensor3d_t& input) const
        {
                assert(n_idims() == input.n_dim1());
                assert(n_irows() <= input.n_rows());
                assert(n_icols() <= input.n_cols());

                // convolution output: odata = bias + weight * (idata @ kdata)
                m_idata = input;

                for (size_t o = 0; o < n_odims(); o ++)
                {
                        const matrix_t& kdata = m_kdata(o);
                        matrix_t& odata = m_odata(o);

                        odata.setConstant(bias(o));

                        for (size_t i = 0; i < n_idims(); i ++)
                        {
                                const matrix_t& idata = m_idata(i);
                                matrix_t& xdata = m_xdata(o, i);

                                switch (m_kdata.n_cols())
                                {
                                case 4:  math::conv_dot<false>(idata, kdata, xdata, math::dot<4, scalar_t>); break;
                                case 8:  math::conv_dot<false>(idata, kdata, xdata, math::dot<8, scalar_t>); break;
                                case 12: math::conv_dot<false>(idata, kdata, xdata, math::dot<12, scalar_t>); break;
                                case 16: math::conv_dot<false>(idata, kdata, xdata, math::dot<16, scalar_t>); break;
                                default: (kmod4x() ?
                                         math::conv_dot<false>(idata, kdata, xdata, math::dot_mod4x<scalar_t>) :
                                         math::conv_dot<false>(idata, kdata, xdata, math::dot_mod4<scalar_t>));
                                         break;
                                }
                                odata.noalias() += weight(o, i) * xdata;
                        }
                }

                return m_odata;
        }

        /////////////////////////////////////////////////////////////////////////////////////////

        const tensor3d_t& conv_layer_t::backward(const tensor3d_t& gradient) const
        {
                assert(n_odims() == gradient.n_dim1());
                assert(n_orows() == gradient.n_rows());
                assert(n_ocols() == gradient.n_cols());

                // convolution gradient
                for (size_t o = 0; o < n_odims(); o ++)
                {
                        const matrix_t& gdata = gradient(o);
                        matrix_t& gkdata = m_gkdata(o);

                        gkdata.setZero();
                        gbias(o) = gdata.sum();

                        for (size_t i = 0; i < n_idims(); i ++)
                        {
                                const matrix_t& idata = m_idata(i);
                                const matrix_t& xdata = m_xdata(o, i);
                                const scalar_t w = weight(o, i);

                                gweight(o, i) = gdata.cwiseProduct(xdata).sum();
                                switch (m_odata.n_cols())
                                {
                                case 4:  math::wconv_dot<true>(idata, gdata, w, gkdata, math::dot<4, scalar_t>); break;
                                case 8:  math::wconv_dot<true>(idata, gdata, w, gkdata, math::dot<8, scalar_t>); break;
                                case 12: math::wconv_dot<true>(idata, gdata, w, gkdata, math::dot<12, scalar_t>); break;
                                case 16: math::wconv_dot<true>(idata, gdata, w, gkdata, math::dot<16, scalar_t>); break;
                                default: (omod4x() ?
                                         math::wconv_dot<true>(idata, gdata, w, gkdata, math::dot_mod4x<scalar_t>) :
                                         math::wconv_dot<true>(idata, gdata, w, gkdata, math::dot_mod4<scalar_t>));
                                         break;
                                }
                        }
                }


                // input gradient
                m_idata.zero();

                for (size_t o = 0; o < n_odims(); o ++)
                {
                        const matrix_t& gdata = gradient(o);
                        const matrix_t& kdata = m_kdata(o);

                        for (size_t i = 0; i < n_idims(); i ++)
                        {
                                matrix_t& idata = m_idata(i);

                                switch (m_kdata.n_cols())
                                {
                                case 4:  backward(gdata, kdata, weight(o, i), idata, math::mad<4, scalar_t>); break;
                                case 8:  backward(gdata, kdata, weight(o, i), idata, math::mad<8, scalar_t>); break;
                                case 12: backward(gdata, kdata, weight(o, i), idata, math::mad<12, scalar_t>); break;
                                case 16: backward(gdata, kdata, weight(o, i), idata, math::mad<16, scalar_t>); break;
                                default: (kmod4x() ?
                                         backward(gdata, kdata, weight(o, i), idata, math::mad_mod4x<scalar_t>) :
                                         backward(gdata, kdata, weight(o, i), idata, math::mad_mod4<scalar_t>));
                                         break;
                                }
                        }
                }

                return m_idata;
        }

        /////////////////////////////////////////////////////////////////////////////////////////

        bool conv_layer_t::save(boost::archive::binary_oarchive& oa) const
        {
                oa << m_params << m_kdata << m_wdata << m_bdata;
                return true;
        }

        /////////////////////////////////////////////////////////////////////////////////////////

        bool conv_layer_t::load(boost::archive::binary_iarchive& ia)
        {
                ia >> m_params >> m_kdata >> m_wdata >> m_bdata;
                return true;
        }

        /////////////////////////////////////////////////////////////////////////////////////////
}


