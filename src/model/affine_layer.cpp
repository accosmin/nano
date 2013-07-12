#include "affine_layer.h"
#include "core/logger.h"
#include "core/string.h"
#include "core/math.h"

namespace ncv
{
        //-------------------------------------------------------------------------------------------------

        static void forward(const matrix_t& idata, const matrix_t& wdata, matrix_t& odata)
        {
                const size_t isize = static_cast<size_t>(idata.size());
                const size_t osize = static_cast<size_t>(odata.size());

                const scalar_t* pwdata = &wdata(0);
                for (size_t o = 0; o < osize; o ++)
                {
                        for (size_t i = 0; i < isize; i ++)
                        {
                                odata(o) += idata(i) * *(pwdata ++);
                        }
                }
        }

        //-------------------------------------------------------------------------------------------------

        static void gradient(const matrix_t& idata, const matrix_t& ogdata, matrix_t& gdata)
        {
                const size_t crows = math::cast<size_t>(gdata.rows());
                const size_t ccols = math::cast<size_t>(gdata.cols());

                const size_t orows = math::cast<size_t>(ogdata.rows());
                const size_t ocols = math::cast<size_t>(ogdata.cols());

                for (size_t r = 0; r < crows; r ++)
                {
                        for (size_t c = 0; c < ccols; c ++)
                        {
                                gdata(r, c) += idata.block(r, c, orows, ocols).cwiseProduct(ogdata).sum();
                        }
                }
        }

        //-------------------------------------------------------------------------------------------------

        static void backward(const matrix_t& ogdata, const matrix_t& cdata, matrix_t& igdata)
        {
                const size_t crows = math::cast<size_t>(cdata.rows());
                const size_t ccols = math::cast<size_t>(cdata.cols());

                const size_t orows = math::cast<size_t>(ogdata.rows());
                const size_t ocols = math::cast<size_t>(ogdata.cols());

                // TODO: this takes 50% of time!

                for (size_t r = 0; r < orows; r ++)
                {
                        for (size_t c = 0; c < ocols; c ++)
                        {
                                igdata.block(r, c, crows, ccols).noalias() += ogdata(r, c) * cdata;
                        }
                }
        }

        //-------------------------------------------------------------------------------------------------

        affine_layer_t::affine_layer_t(
                size_t inputs, size_t irows, size_t icols,
                size_t outputs, size_t crows, size_t ccols,
                const string_t& activation)
        {
                resize(inputs, irows, icols, outputs, crows, ccols, activation);
        }

        //-------------------------------------------------------------------------------------------------

        size_t affine_layer_t::resize(
                size_t inputs, size_t irows, size_t icols,
                size_t outputs, size_t orows, size_t ocols,
                const string_t& activation)
        {
                m_activation = activation;
                m_afunc = activation_manager_t::instance().get(activation, "");
                if (    !m_afunc &&
                        inputs != 0 && irows != 0 && icols != 0 && outputs != 0 && orows != 0 && ocols != 0)
                {
                        log_warning() << "affine layer: invalid activation function ("
                                      << activation << ") out of ("
                                      << text::concatenate(activation_manager_t::instance().ids(), ", ")
                                      << ")";
                        return 0;
                }

                m_idata.resize(inputs, irows, icols);
                m_wdata.resize(outputs, inputs, irows * icols, orows * ocols);
                m_gdata.resize(outputs, inputs, irows * icols, orows * ocols);
                m_odata.resize(outputs, orows, ocols);

                return m_wdata.size();
        }

        //-------------------------------------------------------------------------------------------------

        void affine_layer_t::zero()
        {
                m_wdata.zero();
                zero_grad();
        }

        //-------------------------------------------------------------------------------------------------

        void affine_layer_t::random(scalar_t min, scalar_t max)
        {
                m_wdata.random(min, max);
                zero_grad();
        }

        //-------------------------------------------------------------------------------------------------

        void affine_layer_t::zero_grad() const
        {
                m_gdata.zero();
        }

        //-------------------------------------------------------------------------------------------------

        const tensor3d_t& affine_layer_t::forward(const tensor3d_t& input) const
        {
                throw std::runtime_error("affine_layer_t::forward() - not implemented!");

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
                                const matrix_t& cdata = m_wdata(o, i);

                                ncv::forward(idata, cdata, odata);
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

        const tensor3d_t& affine_layer_t::backward(const tensor3d_t& gradient) const
        {
                throw std::runtime_error("affine_layer_t::backward() - not implemented!");

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
                                const matrix_t& cdata = m_wdata(o, i);
                                matrix_t& igdata = m_idata(i);

                                ncv::backward(ogdata, cdata, igdata);
                        }
                }

                return m_idata;
        }

        //-------------------------------------------------------------------------------------------------

        size_t affine_layer_t::make_network(
                size_t idims, size_t irows, size_t icols,
                const affine_network_params_t& network_params,
                size_t odims,
                affine_network_t& network)
        {
                size_t n_params = 0;

                const size_t n_layers = network_params.size();

                // create hidden layers
                network.clear();
                for (size_t l = 0; l < n_layers; l ++)
                {
                        const affine_layer_param_t& param = network_params[l];

                        const size_t odims = param.m_dims;
                        const size_t orows = param.m_rows;
                        const size_t ocols = param.m_cols;
                        const string_t aid = param.m_activation;

                        affine_layer_t layer;
                        const size_t n_new_params = layer.resize(idims, irows, icols, odims, orows, ocols, aid);

                        if (n_new_params < 1)
                        {
                                log_error() << "invalid affine network description for the layer ["
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
                affine_layer_t layer;
                n_params += layer.resize(idims, irows, icols, odims, 1, 1, "unit");
                network.push_back(layer);

                return n_params;
        }

        //-------------------------------------------------------------------------------------------------

        void affine_layer_t::print_network(const affine_network_t& network)
        {
                for (size_t l = 0; l < network.size(); l ++)
                {
                        const affine_layer_t& layer = network[l];

                        log_info() << "affine network: layer ["
                                   << (l + 1) << "/" << network.size() << "] " << layer.m_activation << " ("
                                   << layer.n_inputs() << "x" << layer.n_irows() << "x" << layer.n_icols()
                                   << " * "
                                   << layer.m_wdata.n_dim1() << "x" << layer.m_wdata.n_dim2() << "x"
                                   << layer.m_wdata.n_rows() << "x" << layer.m_wdata.n_cols()
                                   << ") -> "
                                   << layer.n_outputs() << "x" << layer.n_orows() << "x" << layer.n_ocols()
                                   << ".";
                }
        }

        //-------------------------------------------------------------------------------------------------

        const tensor3d_t& affine_layer_t::forward(const tensor3d_t& _input, const affine_network_t& network)
        {
                const tensor3d_t* input = &_input;
                for (affine_network_t::const_iterator it = network.begin(); it != network.end(); ++ it)
                {
                        input = &it->forward(*input);
                }

                return *input;
        }

        //-------------------------------------------------------------------------------------------------

        void affine_layer_t::backward(const tensor3d_t& _gradient, const affine_network_t& network)
        {
                const tensor3d_t* gradient = &_gradient;
                for (affine_network_t::const_reverse_iterator it = network.rbegin(); it != network.rend(); ++ it)
                {
                        gradient = &it->backward(*gradient);
                }
        }

        //-------------------------------------------------------------------------------------------------
}

