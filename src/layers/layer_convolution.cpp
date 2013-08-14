#include "layer_convolution.h"
#include "core/logger.h"
#include "core/text.h"
#include "core/cast.h"
#include "core/convolution.h"
#include "core/for_each.h"
#include "core/clamp.h"

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

        conv_layer_t::conv_layer_t(const string_t& params)
                :       m_params(params)
        {
        }

        //-------------------------------------------------------------------------------------------------

        size_t conv_layer_t::resize(size_t idims, size_t irows, size_t icols)
        {
                const size_t odims = math::clamp(text::from_params<size_t>(m_params, "convs", 16), 1, 256);
                const size_t crows = math::clamp(text::from_params<size_t>(m_params, "rows", 8), 1, 32);
                const size_t ccols = math::clamp(text::from_params<size_t>(m_params, "cols", 8), 1, 32);

                if (    /*idims < 1 || irows < 1 || icols < 1 ||
                        convs < 1 || crows < 1 || ccols < 1 ||*/
                        irows < crows || icols < ccols)
                {
                        const string_t message =
                                "invalid size (" + text::to_string(idims) + "x" + text::to_string(irows) +
                                 "x" + text::to_string(icols) + ") -> (" + text::to_string(odims) + "x" +
                                 text::to_string(crows) + "x" + text::to_string(ccols) + ")";

                        log_error() << "convolution layer: " << message;
                        throw std::runtime_error("convolution layer: " + message);
                }

                m_idata.resize(idims, irows, icols);
                m_kdata.resize(odims, idims, crows, ccols);
                m_gdata.resize(odims, idims, crows, ccols);
                m_odata.resize(odims, irows - crows + 1, icols - ccols + 1);

                return m_kdata.size();
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

        serializer_t& conv_layer_t::save_params(serializer_t& s) const
        {
                return s << m_kdata;
        }

        //-------------------------------------------------------------------------------------------------

        serializer_t& conv_layer_t::save_grad(serializer_t& s) const
        {
                return s << m_gdata;
        }

        //-------------------------------------------------------------------------------------------------

        deserializer_t& conv_layer_t::load_params(deserializer_t& s) const
        {
                return s >> m_kdata;
        }

        //-------------------------------------------------------------------------------------------------

        bool conv_layer_t::save(boost::archive::binary_oarchive& oa) const
        {
                // TODO

                return false;
        }

        //-------------------------------------------------------------------------------------------------

        bool conv_layer_t::load(boost::archive::binary_iarchive& ia)
        {
                // TODO

                return false;
        }

        //-------------------------------------------------------------------------------------------------

        const tensor3d_t& conv_layer_t::forward(const tensor3d_t& input) const
        {
                assert(n_idims() == input.n_dim1());
                assert(n_irows() <= input.n_rows());
                assert(n_icols() <= input.n_cols());

                // convolution output
                m_idata = input;
                for (size_t o = 0; o < n_odims(); o ++)
                {
                        matrix_t& odata = m_odata(o);
                        odata.setZero();

                        for (size_t i = 0; i < n_idims(); i ++)
                        {
                                const matrix_t& idata = m_idata(i);
                                const matrix_t& kdata = m_kdata(o, i);

                                math::conv_add_dynamic(idata, kdata, odata);
                        }
                }

                return m_odata;
        }

        //-------------------------------------------------------------------------------------------------

        const tensor3d_t& conv_layer_t::backward(const tensor3d_t& gradient) const
        {
                assert(n_odims() == gradient.n_dim1());
                assert(n_orows() == gradient.n_rows());
                assert(n_ocols() == gradient.n_cols());

                // convolution gradient
                m_odata = gradient;
                for (size_t o = 0; o < n_odims(); o ++)
                {
                        const matrix_t& odata = m_odata(o);

                        for (size_t i = 0; i < n_idims(); i ++)
                        {
                                const matrix_t& idata = m_idata(i);
                                matrix_t& gdata = m_gdata(o, i);

                                math::conv_add_dynamic(idata, odata, gdata);
                        }
                }

                // input gradient
                m_idata.zero();
                for (size_t o = 0; o < n_odims(); o ++)
                {
                        const matrix_t& odata = m_odata(o);

                        for (size_t i = 0; i < n_idims(); i ++)
                        {
                                const matrix_t& kdata = m_kdata(o, i);
                                matrix_t& idata = m_idata(i);

                                ncv::backward(odata, kdata, idata);
                        }
                }

                return m_idata;
        }

        //-------------------------------------------------------------------------------------------------
}

