#include "conv_layer.h"
#include "core/logger.h"
#include "core/text.h"
#include "core/cast.h"
#include "core/convolution.h"
#include "core/for_each.h"

namespace ncv
{
        //-------------------------------------------------------------------------------------------------

//        static void backward(const matrix_t& ogdata, const matrix_t& kdata, matrix_t& igdata)
//        {
//                const size_t krows = math::cast<size_t>(kdata.rows());
//                const size_t kcols = math::cast<size_t>(kdata.cols());

//                const size_t orows = math::cast<size_t>(ogdata.rows());
//                const size_t ocols = math::cast<size_t>(ogdata.cols());

////                TODO: move this part to core/convolution.h -> deconv_add()

//                for (size_t r = 0; r < orows; r ++)
//                {
//                        for (size_t c = 0; c < ocols; c ++)
//                        {
//                                igdata.block(r, c, krows, kcols).noalias() += ogdata(r, c) * kdata;

//                        }
//                }
//        }

        static void backward(const matrix_t& ogdata, const matrix_t& kdata, matrix_t& igdata)
        {
                const int krows = math::cast<int>(kdata.rows());
                const int kcols = math::cast<int>(kdata.cols());
                const int kcols4 = kcols - (kcols % 4);

                const int orows = math::cast<int>(ogdata.rows());
                const int ocols = math::cast<int>(ogdata.cols());

//                TODO: move this part to core/convolution.h -> deconv_add()

//                for (int r = 0; r < orows; r ++)
//                {
//                        const scalar_t* pogdata = &ogdata(r, 0);

//                        for (int kr = 0; kr < krows; kr ++)
//                        {
//                                const scalar_t* pkdata = &kdata(kr, 0);
//                                scalar_t* pigdata = &igdata(r + kr);

//                                for (int c = 0; c < ocols; c ++)
//                                {
////                                        for (int kc = 0; kc < kcols4; kc += 4)
////                                        {
////                                                pigdata[c + kc + 0] += pogdata[c] * pkdata[kc + 0];
////                                                pigdata[c + kc + 1] += pogdata[c] * pkdata[kc + 1];
////                                                pigdata[c + kc + 2] += pogdata[c] * pkdata[kc + 2];
////                                                pigdata[c + kc + 3] += pogdata[c] * pkdata[kc + 3];
////                                        }
////                                        for (int kc = kcols4; kc < kcols; kc ++)
////                                        {
////                                                pigdata[c + kc + 0] += pogdata[c] * pkdata[kc];
////                                        }

//                                        for (int kc = 0; kc < kcols; kc ++)
//                                        {
//                                                pigdata[c + kc] += pogdata[c] * pkdata[kc];
//                                        }
//                                }
//                        }
//                }

                for (int r = 0; r < orows; r ++)
                {
                        const scalar_t* pogdata = &ogdata(r, 0);

                        for (int kr = 0; kr < krows; kr ++)
                        {
                                const scalar_t* pkdata = &kdata(kr, 0);
                                scalar_t* pigdata = &igdata(r + kr, 0);

                                for (int c = 0; c < ocols; c ++)
                                {
//                                        for (int kc = 0; kc < kcols; kc ++)
//                                        {
//                                                pigdata[c + kc] += pogdata[c] * pkdata[kc];
//                                        }

                                        for (int kc = 0; kc < kcols4; kc += 4)
                                        {
                                                pigdata[c + kc + 0] += pogdata[c] * pkdata[kc + 0];
                                                pigdata[c + kc + 1] += pogdata[c] * pkdata[kc + 1];
                                                pigdata[c + kc + 2] += pogdata[c] * pkdata[kc + 2];
                                                pigdata[c + kc + 3] += pogdata[c] * pkdata[kc + 3];
                                        }
                                        for (int kc = kcols4; kc < kcols; kc ++)
                                        {
                                                pigdata[c + kc + 0] += pogdata[c] * pkdata[kc];
                                        }
                                }
                        }
                }
        }

        //-------------------------------------------------------------------------------------------------

        conv_layer_t::conv_layer_t(
                size_t inputs, size_t irows, size_t icols,
                size_t outputs, size_t crows, size_t ccols,
                const string_t& activation)
        {
                resize(inputs, irows, icols, outputs, crows, ccols, activation);
        }

        //-------------------------------------------------------------------------------------------------

        size_t conv_layer_t::resize(
                size_t inputs, size_t irows, size_t icols,
                size_t outputs, size_t crows, size_t ccols,
                const string_t& activation)
        {
                if (    /*inputs < 1 || irows < 1 || icols < 1 ||
                        outputs < 1 || crows < 1 || ccols < 1 ||*/
                        irows < crows || icols < ccols)
                {
                        const string_t message =
                                "invalid size (" + text::to_string(inputs) + "x" + text::to_string(irows) +
                                 "x" + text::to_string(icols) + ") -> (" + text::to_string(outputs) + "x" +
                                 text::to_string(crows) + "x" + text::to_string(ccols) + ")";

                        log_warning() << "convolution layer: " << message;
                        throw std::runtime_error("convolution layer: " + message);
                }

                m_activation = activation;
                m_idata.resize(inputs, irows, icols);
                m_kdata.resize(outputs, inputs, crows, ccols);
                m_gdata.resize(outputs, inputs, crows, ccols);
                m_odata.resize(outputs, irows - crows + 1, icols - ccols + 1);

                set_activation();

                return m_kdata.size();
        }

        //-------------------------------------------------------------------------------------------------

        void conv_layer_t::set_activation()
        {
                m_afunc = activation_manager_t::instance().get(m_activation);
                if (    !m_afunc &&
                        n_idims() != 0 && n_irows() != 0 && n_icols() != 0 &&
                        n_odims() != 0)
                {
                        const string_t message =
                                "invalid activation function (" + m_activation + ") out of (" +
                                text::concatenate(activation_manager_t::instance().ids(), ", ") + ")";

                        log_warning() << "convolution layer: " << message;
                        throw std::runtime_error("convolution layer: " + message);
                }
        }

        //-------------------------------------------------------------------------------------------------

        void conv_layer_t::zero_params()
        {
                m_kdata.zero();
                zero_grad();
        }

        //-------------------------------------------------------------------------------------------------

        void conv_layer_t::random_params(scalar_t min, scalar_t max)
        {
                m_kdata.random(min, max);
                zero_grad();
        }

        //-------------------------------------------------------------------------------------------------

        void conv_layer_t::zero_grad() const
        {
                m_gdata.zero();
        }

        //-------------------------------------------------------------------------------------------------

        const tensor3d_t& conv_layer_t::forward(const tensor3d_t& input) const
        {
                assert(n_idims() == input.n_dim1());
                assert(n_irows() <= input.n_rows());
                assert(n_icols() <= input.n_cols());
                assert(m_afunc);

                m_idata = input;

                // outputs
                const activation_t& afunc = *m_afunc;
                for (size_t o = 0; o < n_odims(); o ++)
                {
                        matrix_t& odata = m_odata(o);
                        odata.setZero();

                        // convolution output
                        for (size_t i = 0; i < n_idims(); i ++)
                        {
                                const matrix_t& idata = m_idata(i);
                                const matrix_t& kdata = m_kdata(o, i);

                                math::conv_add_dynamic(idata, kdata, odata);
                        }

                        // activation
                        math::for_each(odata, [&] (scalar_t& v) { v = afunc.value(v); });
                }

                return m_odata;
        }

        //-------------------------------------------------------------------------------------------------

        const tensor3d_t& conv_layer_t::backward(const tensor3d_t& gradient) const
        {
                assert(n_odims() == gradient.n_dim1());
                assert(n_orows() == gradient.n_rows());
                assert(n_ocols() == gradient.n_cols());
                assert(m_afunc);

                // outputs
                const activation_t& afunc = *m_afunc;
                for (size_t o = 0; o < n_odims(); o ++)
                {
                        const matrix_t& gdata = gradient(o);
                        const matrix_t& ogdata = m_odata(o);
                        matrix_t& odata = m_odata(o);

                        // activation
                        const size_t size = math::cast<size_t>(odata.size());
                        for (size_t ii = 0; ii < size; ii ++)
                        {
                                odata(ii) = gdata(ii) * afunc.vgrad(odata(ii));
                        }

                        // convolution gradient
                        for (size_t i = 0; i < n_idims(); i ++)
                        {
                                const matrix_t& idata = m_idata(i);
                                matrix_t& gdata = m_gdata(o, i);

                                math::conv_add_dynamic(idata, ogdata, gdata);
                        }
                }

                // input gradient
                m_idata.zero();
                for (size_t o = 0; o < n_odims(); o ++)
                {
                        const matrix_t& ogdata = m_odata(o);

                        for (size_t i = 0; i < n_idims(); i ++)
                        {
                                const matrix_t& kdata = m_kdata(o, i);
                                matrix_t& igdata = m_idata(i);

                                ncv::backward(ogdata, kdata, igdata);
                        }
                }

                return m_idata;
        }

        //-------------------------------------------------------------------------------------------------

        void conv_layer_t::backward(const conv_layer_t& layer) const
        {
                assert(n_idims() == layer.n_idims());
                assert(n_odims() == layer.n_odims());
                assert(n_irows() == layer.n_irows());
                assert(n_icols() == layer.n_icols());
                assert(n_orows() == layer.n_orows());
                assert(n_ocols() == layer.n_ocols());

                for (size_t o = 0; o < n_odims(); o ++)
                {
                        for (size_t i = 0; i < n_idims(); i ++)
                        {
                                m_gdata(o, i) += layer.m_gdata(o, i);
                        }
                }
        }

        //-------------------------------------------------------------------------------------------------
}

