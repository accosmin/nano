#include "conv_layer.h"
#include "core/logger.h"
#include "core/string.h"
#include "core/math.h"

namespace ncv
{
        //-------------------------------------------------------------------------------------------------

        static void forward(const matrix_t& idata, const matrix_t& kdata, matrix_t& odata)
        {
                const size_t krows = math::cast<size_t>(kdata.rows());
                const size_t kcols = math::cast<size_t>(kdata.cols());

                const size_t orows = math::cast<size_t>(odata.rows());
                const size_t ocols = math::cast<size_t>(odata.cols());

                for (size_t r = 0; r < orows; r ++)
                {
                        for (size_t c = 0; c < ocols; c ++)
                        {
                                odata(r, c) += idata.block(r, c, krows, kcols).cwiseProduct(kdata).sum();
                        }
                }
        }

        //-------------------------------------------------------------------------------------------------

        static void gradient(const matrix_t& idata, const matrix_t& ogdata, matrix_t& gdata)
        {
                const size_t krows = math::cast<size_t>(gdata.rows());
                const size_t kcols = math::cast<size_t>(gdata.cols());

                const size_t orows = math::cast<size_t>(ogdata.rows());
                const size_t ocols = math::cast<size_t>(ogdata.cols());

                for (size_t r = 0; r < krows; r ++)
                {
                        for (size_t c = 0; c < kcols; c ++)
                        {
                                gdata(r, c) += idata.block(r, c, orows, ocols).cwiseProduct(ogdata).sum();
                        }
                }
        }

        //-------------------------------------------------------------------------------------------------

        static void backward(const matrix_t& ogdata, const matrix_t& kdata, matrix_t& igdata)
        {
                const size_t krows = math::cast<size_t>(kdata.rows());
                const size_t kcols = math::cast<size_t>(kdata.cols());

                const size_t orows = math::cast<size_t>(ogdata.rows());
                const size_t ocols = math::cast<size_t>(ogdata.cols());

                // TODO: this takes 50% of time!

                for (size_t r = 0; r < orows; r ++)
                {
                        for (size_t c = 0; c < ocols; c ++)
                        {
                                igdata.block(r, c, krows, kcols).noalias() += ogdata(r, c) * kdata;
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
                        n_inputs() != 0 && n_irows() != 0 && n_icols() != 0 &&
                        n_outputs() != 0)
                {
                        const string_t message =
                                "invalid activation function (" + m_activation + ") out of (" +
                                text::concatenate(activation_manager_t::instance().ids(), ", ") + ")";

                        log_warning() << "convolution layer: " << message;
                        throw std::runtime_error("convolution layer: " + message);
                }
        }

        //-------------------------------------------------------------------------------------------------

        void conv_layer_t::zero()
        {
                m_kdata.zero();
        }

        //-------------------------------------------------------------------------------------------------

        void conv_layer_t::random(scalar_t min, scalar_t max)
        {
                m_kdata.random(min, max);
        }

        //-------------------------------------------------------------------------------------------------

        const tensor3d_t& conv_layer_t::forward(const tensor3d_t& input) const
        {
                assert(n_inputs() == input.n_dim1());
                assert(n_irows() <= input.n_rows());
                assert(n_icols() <= input.n_cols());
                assert(m_afunc);

                m_idata = input;

                const activation_t& afunc = *m_afunc;

                // output
                m_odata.zero();
                for (size_t o = 0; o < n_outputs(); o ++)
                {
                        matrix_t& odata = m_odata(o);

                        for (size_t i = 0; i < n_inputs(); i ++)
                        {
                                const matrix_t& idata = m_idata(i);
                                const matrix_t& kdata = m_kdata(o, i);

                                ncv::forward(idata, kdata, odata);
                        }
                }

                // activation
                for (size_t o = 0; o < n_outputs(); o ++)
                {
                        matrix_t& odata = m_odata(o);

                        math::for_each(odata, [&] (scalar_t& v) { v = afunc.value(v); });
                }

                return m_odata;
        }

        //-------------------------------------------------------------------------------------------------

        const tensor3d_t& conv_layer_t::backward(const tensor3d_t& gradient) const
        {
                assert(n_outputs() == gradient.n_dim1());
                assert(n_orows() == gradient.n_rows());
                assert(n_ocols() == gradient.n_cols());
                assert(m_afunc);

                const activation_t& afunc = *m_afunc;

                // activation
                for (size_t o = 0; o < n_outputs(); o ++)
                {
                        const matrix_t& gdata = gradient(o);
                        matrix_t& odata = m_odata(o);

                        const size_t size = math::cast<size_t>(odata.size());
                        for (size_t ii = 0; ii < size; ii ++)
                        {
                                odata(ii) = gdata(ii) * afunc.vgrad(odata(ii));
                        }
                }

                // convolution gradient
                m_gdata.zero();
                for (size_t o = 0; o < n_outputs(); o ++)
                {
                        const matrix_t& ogdata = m_odata(o);

                        for (size_t i = 0; i < n_inputs(); i ++)
                        {
                                const matrix_t& idata = m_idata(i);
                                matrix_t& gdata = m_gdata(o, i);

                                ncv::gradient(idata, ogdata, gdata);
                        }
                }

                // input gradient
                m_idata.zero();
                for (size_t o = 0; o < n_outputs(); o ++)
                {
                        const matrix_t& ogdata = m_odata(o);

                        for (size_t i = 0; i < n_inputs(); i ++)
                        {
                                const matrix_t& kdata = m_kdata(o, i);
                                matrix_t& igdata = m_idata(i);

                                ncv::backward(ogdata, kdata, igdata);
                        }
                }

                return m_idata;
        }

        //-------------------------------------------------------------------------------------------------
}

