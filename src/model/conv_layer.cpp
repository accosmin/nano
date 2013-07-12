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
                        log_warning() << "convolution layer: invalid size ("
                                      << inputs << "x" << irows << "x" << icols
                                      << ") -> (" << outputs << "x" << crows << "x" << ccols << ")";
                        return 0;
                }

                m_activation = activation;
                m_afunc = activation_manager_t::instance().get(activation, "");
                if (    !m_afunc &&
                        inputs != 0 && irows != 0 && icols != 0 && outputs != 0 && crows != 0 && ccols != 0)
                {
                        log_warning() << "convolution layer: invalid activation function ("
                                      << activation << ") out of ("
                                      << text::concatenate(activation_manager_t::instance().ids(), ", ")
                                      << ")";
                        return 0;
                }

                m_idata.resize(inputs, irows, icols);
                m_kdata.resize(outputs, inputs, crows, ccols);
                m_gdata.resize(outputs, inputs, crows, ccols);
                m_odata.resize(outputs, irows - crows + 1, icols - ccols + 1);

                return m_kdata.size();
        }

        //-------------------------------------------------------------------------------------------------

        void conv_layer_t::zero()
        {
                m_kdata.zero();
                zero_grad();
        }

        //-------------------------------------------------------------------------------------------------

        void conv_layer_t::random(scalar_t min, scalar_t max)
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
                assert(afunc);

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

        size_t conv_layer_t::make_network(
                size_t idims, size_t irows, size_t icols,
                const conv_network_params_t& network_params,
                size_t odims,
                conv_network_t& network)
        {
                size_t n_params = 0;

                const size_t n_layers = network_params.size();

                // create hidden layers
                network.clear();
                for (size_t l = 0; l < n_layers; l ++)
                {
                        const conv_layer_param_t& param = network_params[l];

                        const size_t convs = param.m_convs;
                        const size_t crows = param.m_crows;
                        const size_t ccols = param.m_ccols;
                        const string_t aid = param.m_activation;

                        conv_layer_t layer;
                        const size_t n_new_params = layer.resize(idims, irows, icols, convs, crows, ccols, aid);

                        if (n_new_params < 1)
                        {
                                log_error() << "invalid convolution network description for the layer ["
                                            << (l + 1) << "/" << n_layers
                                            << "(" << idims << ":" << irows << "x" << icols << ")!";
                                return 0;
                        }

                        n_params += n_new_params;
                        network.push_back(layer);

                        idims = layer.n_outputs();
                        irows = layer.n_orows();
                        icols = layer.n_ocols();
                }

                // create output layer
                conv_layer_t layer;
                n_params += layer.resize(idims, irows, icols, odims, irows, icols, "unit");
                network.push_back(layer);

                return n_params;
        }

        //-------------------------------------------------------------------------------------------------

        void conv_layer_t::print_network(const conv_network_t& network)
        {
                for (size_t l = 0; l < network.size(); l ++)
                {
                        const conv_layer_t& layer = network[l];

                        log_info() << "convolution network: layer ["
                                   << (l + 1) << "/" << network.size() << "] " << layer.m_activation << " ("
                                   << layer.n_inputs() << "x" << layer.n_irows() << "x" << layer.n_icols()
                                   << " * "
                                   << layer.m_kdata.n_dim1() << "x" << layer.m_kdata.n_dim2() << "x"
                                   << layer.m_kdata.n_rows() << "x" << layer.m_kdata.n_cols()
                                   << ") -> "
                                   << layer.n_outputs() << "x" << layer.n_orows() << "x" << layer.n_ocols()
                                   << ".";
                }
        }

        //-------------------------------------------------------------------------------------------------

        const tensor3d_t& conv_layer_t::forward(const tensor3d_t& _input, const conv_network_t& network)
        {
                const tensor3d_t* input = &_input;
                for (conv_network_t::const_iterator it = network.begin(); it != network.end(); ++ it)
                {
                        input = &it->forward(*input);
                }

                return *input;
        }

        //-------------------------------------------------------------------------------------------------

        void conv_layer_t::backward(const tensor3d_t& _gradient, const conv_network_t& network)
        {
                const tensor3d_t* gradient = &_gradient;
                for (conv_network_t::const_reverse_iterator it = network.rbegin(); it != network.rend(); ++ it)
                {
                        gradient = &it->backward(*gradient);
                }
        }

        //-------------------------------------------------------------------------------------------------
}

