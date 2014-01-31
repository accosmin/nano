#include "layer_convolution.h"
#include "text.h"
#include "vectorizer.h"
#include "image.h"
#include "util/logger.h"
#include "util/math.hpp"
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
                                const scalar_t w = weight(o, i);
                                matrix_t& xdata = m_xdata(o, i);

                                xdata.setZero();
                                math::conv_dot(idata, kdata, xdata);
                                odata.noalias() += w * xdata;
                        }
                }

                return m_odata;
        }

        /////////////////////////////////////////////////////////////////////////////////////////

        template
        <
                typename tmad
        >
        void _backward(const matrix_t& ogdata, const matrix_t& kdata, scalar_t weight, matrix_t& igdata, tmad madop)
        {
                for (auto r = 0; r < ogdata.rows(); r ++)
                {
                        const scalar_t* pogdata = &ogdata(r, 0);

                        for (auto kr = 0; kr < kdata.rows(); kr ++)
                        {
                                const scalar_t* pkdata = &kdata(kr, 0);
                                scalar_t* pigdata = &igdata(r + kr, 0);

                                for (auto c = 0; c < ogdata.cols(); c ++)
                                {
                                        const scalar_t w = weight * pogdata[c];

                                        madop(pkdata, w, pigdata + c, kdata.cols());
                                }
                        }
                }
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
                                math::wconv_dot(idata, gdata, w, gkdata);
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
                                const scalar_t w = weight(o, i);

                                const auto kcols = m_kdata.n_cols();
                                if (kcols == 3) _backward(gdata, kdata, w, idata, math::mad<3, scalar_t>);
                                else if (kcols == 4) _backward(gdata, kdata, w, idata, math::mad<4, scalar_t>);
                                else if (kcols == 5) _backward(gdata, kdata, w, idata, math::mad<5, scalar_t>);
                                else if (kcols == 6) _backward(gdata, kdata, w, idata, math::mad<6, scalar_t>);
                                else if (kcols == 7) _backward(gdata, kdata, w, idata, math::mad<7, scalar_t>);
                                else if (kcols == 8) _backward(gdata, kdata, w, idata, math::mad<8, scalar_t>);
                                else if (kcols == 9) _backward(gdata, kdata, w, idata, math::mad<9, scalar_t>);
                                else if (kcols == 10) _backward(gdata, kdata, w, idata, math::mad<10, scalar_t>);
                                else if (kcols == 11) _backward(gdata, kdata, w, idata, math::mad<11, scalar_t>);
                                else if (kcols == 12) _backward(gdata, kdata, w, idata, math::mad<12, scalar_t>);
                                else if (kcols == 13) _backward(gdata, kdata, w, idata, math::mad<13, scalar_t>);
                                else if (kcols == 14) _backward(gdata, kdata, w, idata, math::mad<14, scalar_t>);
                                else if (kcols == 15) _backward(gdata, kdata, w, idata, math::mad<15, scalar_t>);
                                else if ((kcols & 3) == 0) _backward(gdata, kdata, w, idata, math::mad_mod4<scalar_t>);
                                else _backward(gdata, kdata, w, idata, math::mad_mod4x<scalar_t>);
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

        bool conv_layer_t::save_as_image(const string_t& basepath) const
        {
                const size_t gcols = 8;
                const size_t grows = (n_kdims() + gcols / 2) / gcols;

                const size_t border = std::max(size_t(4), (n_krows() + n_kcols()) / 4);
                const size_t radius = 1;
                const size_t rows = n_krows() * grows + border * (grows + 1);
                const size_t cols = n_kcols() * gcols + border * (gcols + 1);

                const rgba_t back_color = color::make_rgba(225, 225, 0);
                const rgba_t border_color = color::make_rgba(0, 225, 0);

                rgba_matrix_t rgba(rows, cols);
                rgba.setConstant(back_color);

                // compose an image from all convolution kernels ...
                for (size_t k = 0, r = 0; r < grows; r ++)
                {
                        for (size_t c = 0; c < gcols; c ++, k ++)
                        {
                                if (k < n_kdims())
                                {
                                        const matrix_t& kdata = m_kdata(k);
                                        const rgba_matrix_t kimage = color::make_rgba_sign(kdata);

                                        const size_t iy = n_krows() * r + border * (r + 1);
                                        const size_t ix = n_kcols() * c + border * (c + 1);
                                        const size_t ih = n_krows();
                                        const size_t iw = n_kcols();

                                        // kernel border
                                        rgba.block(iy - radius,
                                                   ix - radius,
                                                   ih + radius * 2,
                                                   iw + radius * 2).setConstant(border_color);

                                        // kernel patch
                                        rgba.block(iy, ix, ih, iw) = kimage;
                                }
                        }
                }

                // ... and save it
                const string_t path = basepath + "_conv.png";
                log_info() << "saving images to <" << path << "> ...";
                return ncv::save_rgba(path, rgba);
        }

        /////////////////////////////////////////////////////////////////////////////////////////
}


