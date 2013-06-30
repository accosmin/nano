#include "conv_layer.h"
#include "core/logger.h"
#include "core/string.h"

namespace ncv
{
        //-------------------------------------------------------------------------------------------------

        static void forward(const matrix_t& idata, const matrix_t& cdata, matrix_t& odata)
        {
                const size_t crows = static_cast<size_t>(cdata.rows());
                const size_t ccols = static_cast<size_t>(cdata.cols());

                const size_t orows = static_cast<size_t>(idata.rows() - crows + 1);
                const size_t ocols = static_cast<size_t>(idata.cols() - ccols + 1);

                for (size_t r = 0; r < orows; r ++)
                {
                        for (size_t c = 0; c < ocols; c ++)
                        {
                                odata(r, c) += idata.block(r, c, crows, ccols).cwiseProduct(cdata).sum();
                        }
                }
        }

        //-------------------------------------------------------------------------------------------------

        static void gradient(const matrix_t& idata, const matrix_t& ogdata, matrix_t& gdata)
        {
                const size_t crows = static_cast<size_t>(gdata.rows());
                const size_t ccols = static_cast<size_t>(gdata.cols());

                const size_t orows = static_cast<size_t>(ogdata.rows());
                const size_t ocols = static_cast<size_t>(ogdata.cols());

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
                const size_t crows = static_cast<size_t>(cdata.rows());
                const size_t ccols = static_cast<size_t>(cdata.cols());

                const size_t orows = static_cast<size_t>(ogdata.rows());
                const size_t ocols = static_cast<size_t>(ogdata.cols());

                for (size_t r = 0; r < orows; r ++)
                {
                        for (size_t c = 0; c < ocols; c ++)
                        {
                                igdata.block(r, c, crows, ccols) += ogdata(r, c) * cdata;
                        }
                }
        }

        //-------------------------------------------------------------------------------------------------

        conv_layer_t::conv_layer_t(size_t inputs, size_t irows, size_t icols,
                                   size_t outputs, size_t crows, size_t ccols)
        {
                resize(inputs, irows, icols, outputs, crows, ccols);
        }

        //-------------------------------------------------------------------------------------------------

        size_t conv_layer_t::resize(size_t inputs, size_t irows, size_t icols,
                                    size_t outputs, size_t crows, size_t ccols)
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

                m_idata.resize(inputs, irows, icols);
                m_cdata.resize(outputs, inputs, crows, ccols);
                m_gdata.resize(outputs, inputs, crows, ccols);
                m_odata.resize(outputs, irows - crows + 1, icols - ccols + 1);

                return m_cdata.size();
        }

        //-------------------------------------------------------------------------------------------------

        void conv_layer_t::zero()
        {
                m_cdata.zero();
                zero_grad();
        }

        //-------------------------------------------------------------------------------------------------

        void conv_layer_t::random(scalar_t min, scalar_t max)
        {
                m_cdata.random(min, max);
                zero_grad();
        }

        //-------------------------------------------------------------------------------------------------

        void conv_layer_t::zero_grad()
        {
                m_gdata.zero();
        }

        //-------------------------------------------------------------------------------------------------

        const tensor3d_t& conv_layer_t::forward(const tensor3d_t& input) const
        {
                assert(n_inputs() == input.n_dim1());
                assert(n_irows() <= input.n_rows());
                assert(n_icols() <= input.n_cols());

                m_idata = input;
                m_odata.zero();

                // output
                for (size_t o = 0; o < n_outputs(); o ++)
                {
                        matrix_t& odata = m_odata(o);

                        for (size_t i = 0; i < n_inputs(); i ++)
                        {
                                const matrix_t& idata = m_idata(i);
                                const matrix_t& cdata = m_cdata(o, i);

                                ncv::forward(idata, cdata, odata);
                        }
                }

                return m_odata;
        }

        //-------------------------------------------------------------------------------------------------

        const tensor3d_t& conv_layer_t::backward(const tensor3d_t& gradient)
        {
                assert(n_outputs() == gradient.n_dim1());
                assert(n_orows() == gradient.n_rows());
                assert(n_ocols() == gradient.n_cols());

                // convolution gradient
                for (size_t o = 0; o < n_outputs(); o ++)
                {
                        const matrix_t& ogdata = gradient(o);

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
                        const matrix_t& ogdata = gradient(o);

                        for (size_t i = 0; i < n_inputs(); i ++)
                        {
                                const matrix_t& cdata = m_cdata(o, i);
                                matrix_t& igdata = m_idata(i);

                                ncv::backward(ogdata, cdata, igdata);
                        }
                }

                return m_idata;
        }

        //-------------------------------------------------------------------------------------------------

        size_t conv_layer_t::make_network(
                size_t idims, size_t irows, size_t icols,
                const std::vector<size_t>& network_params, // [#convolutions, #crows, #cols]*
                size_t odims,
                std::vector<conv_layer_t>& network)
        {
                size_t n_params = 0;

                const size_t n_layers = network_params.size() / 3;
                if (network_params.size() % 3 != 0)
                {
                        log_error() << "invalid convolution network description ("
                                    << text::concatenate(network_params) << ")!";
                        return 0;
                }

                // create hidden layers
                network.clear();
                for (size_t l = 0; l < n_layers; l ++)
                {
                        conv_layer_t layer;
                        const size_t convs = network_params[3 * l + 0];
                        const size_t crows = network_params[3 * l + 1];
                        const size_t ccols = network_params[3 * l + 2];

                        const size_t n_new_params = layer.resize(idims, irows, icols, convs, crows, ccols);
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
                n_params += layer.resize(idims, irows, icols, odims, irows, icols);
                network.push_back(layer);

                return n_params;
        }

        //-------------------------------------------------------------------------------------------------

        void conv_layer_t::print_network(const std::vector<conv_layer_t>& network)
        {
                for (size_t l = 0; l < network.size(); l ++)
                {
                        const conv_layer_t& layer = network[l];

                        log_info() << "convolution network: layer ["
                                   << (l + 1) << "/" << network.size() << "] convolutions - "
                                   << layer.m_cdata.n_dim1() << "x" << layer.m_cdata.n_dim2() << " of "
                                   << layer.m_cdata.n_rows() << "x" << layer.m_cdata.n_cols()
                                   << ", inputs "
                                   << layer.n_inputs() << "x" << layer.n_irows() << "x" << layer.n_icols()
                                   << ", outputs "
                                   << layer.n_outputs() << "x" << layer.n_orows() << "x" << layer.n_ocols() << ".";
                }
        }

        //-------------------------------------------------------------------------------------------------
}

