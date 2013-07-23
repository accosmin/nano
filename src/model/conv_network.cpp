#include "conv_network.h"
#include "core/logger.h"

namespace ncv
{
        //-------------------------------------------------------------------------------------------------

        conv_network_t::conv_network_t(const string_t& params)
        {
                // decode the network structure
                const string_t network_desc = text::from_params<string_t>(params, "network", "");
                if (!network_desc.empty())
                {
                        strings_t network_tokens;
                        text::split(network_tokens, network_desc, text::is_any_of(","));

                        if (network_tokens.empty())
                        {
                                throw std::runtime_error("invalid convolution network <" +
                                                         network_desc + ">!");
                        }

                        // layers ...
                        for (size_t l = 0; l < network_tokens.size(); l ++)
                        {
                                if (network_tokens[l].empty())
                                {
                                        continue;
                                }

                                strings_t layer_tokens;
                                text::split(layer_tokens, network_tokens[l], text::is_any_of(":"));

                                if (layer_tokens.size() != 4)
                                {
                                        throw std::runtime_error("invalid layer description <" +
                                                                 network_tokens[l] + "> for the nework <" +
                                                                 network_desc + ">!");
                                }

                                // convolutions ...
                                const size_t convs = text::from_string<size_t>(layer_tokens[0]);
                                const size_t crows = text::from_string<size_t>(layer_tokens[1]);
                                const size_t ccols = text::from_string<size_t>(layer_tokens[2]);
                                const string_t activation = layer_tokens[3];

                                m_params.push_back(conv_layer_param_t(convs, crows, ccols, activation));
                        }
                }
        }

        //-------------------------------------------------------------------------------------------------

        conv_network_t::conv_network_t(const conv_layer_params_t& params)
                :       m_params(params)
        {
        }

        //-------------------------------------------------------------------------------------------------

        vector_t conv_network_t::value(const tensor3d_t& _input) const
        {
                const tensor3d_t* input = &_input;
                for (conv_layers_t::const_iterator it = m_layers.begin(); it != m_layers.end(); ++ it)
                {
                        input = &it->forward(*input);
                }

                const tensor3d_t& output = *input;
                vector_t result(output.size());
                serializer_t(result) << output;

                return result;
        }

        //-------------------------------------------------------------------------------------------------

        void conv_network_t::zero_grad() const
        {
                for (const conv_layer_t& layer : m_layers)
                {
                        layer.zero_grad();
                }
        }

        //-------------------------------------------------------------------------------------------------

        void conv_network_t::cumulate_grad(const vector_t& vgradient) const
        {
                tensor3d_t _gradient(n_outputs(), 1, 1);
                deserializer_t(vgradient) >> _gradient;

                const tensor3d_t* gradient = &_gradient;
                for (conv_layers_t::const_reverse_iterator it = m_layers.rbegin(); it != m_layers.rend(); ++ it)
                {
                        gradient = &it->backward(*gradient);
                }
        }

        //-------------------------------------------------------------------------------------------------

        vector_t conv_network_t::grad() const
        {
                vector_t mgradient(n_parameters());
                serializer_t s(mgradient);
                for (const conv_layer_t& layer : m_layers)
                {
                        s << layer.gdata();
                }

                return mgradient;
        }

        //-------------------------------------------------------------------------------------------------

        bool conv_network_t::save_params(vector_t& x) const
        {
                if (math::cast<size_t>(x.size()) == n_parameters())
                {
                        serializer_t(x) << m_layers;
                        return true;
                }

                else
                {
                        return false;
                }
        }

        //-------------------------------------------------------------------------------------------------

        bool conv_network_t::load_params(const vector_t& x)
        {
                if (math::cast<size_t>(x.size()) == n_parameters())
                {
                        deserializer_t(x) >> m_layers;
                        zero_grad();
                        return true;
                }

                else
                {
                        return false;
                }
        }

        //-------------------------------------------------------------------------------------------------

        void conv_network_t::zero_params()
        {
                for (conv_layer_t& layer : m_layers)
                {
                        layer.zero_params();
                }
        }

        //-------------------------------------------------------------------------------------------------

        void conv_network_t::random_params()
        {
                const scalar_t min = -1.0 / sqrt(n_parameters());
                const scalar_t max = +1.0 / sqrt(n_parameters());

                for (conv_layer_t& layer : m_layers)
                {
                        layer.random_params(min, max);
                }
        }

        //-------------------------------------------------------------------------------------------------

        bool conv_network_t::save(boost::archive::binary_oarchive& oa) const
        {
                oa << m_layers;
                oa << m_params;

                return true;    // fixme: how to check status of the stream?!
        }

        //-------------------------------------------------------------------------------------------------

        bool conv_network_t::load(boost::archive::binary_iarchive& ia)
        {
                ia >> m_layers;
                ia >> m_params;

                return true;    // fixme: how to check status of the stream?!
        }

        //-------------------------------------------------------------------------------------------------

        size_t conv_network_t::resize()
        {
                size_t idims = n_inputs();
                size_t irows = n_rows();
                size_t icols = n_cols();
                size_t odims = n_outputs();
                size_t n_params = 0;

                m_layers.clear();

                // create hidden layers
                for (size_t l = 0; l < m_params.size(); l ++)
                {
                        const conv_layer_param_t& param = m_params[l];

                        const size_t convs = param.m_convs;
                        const size_t crows = param.m_crows;
                        const size_t ccols = param.m_ccols;
                        const string_t aid = param.m_activation;

                        conv_layer_t layer;
                        const size_t n_new_params = layer.resize(idims, irows, icols, convs, crows, ccols, aid);
                        layer.zero_params();

                        if (n_new_params < 1)
                        {
                                const auto message =
                                        boost::format("convolution network: invalid convolution network description "\
                                                      "for the layer [%1%/%2%]: (%3%:%4%x%5%)!")
                                        % (l + 1) % m_params.size() % idims % irows % icols;

                                log_error() << message.str();
                                throw std::runtime_error(message.str());
                        }

                        n_params += n_new_params;
                        m_layers.push_back(layer);

                        idims = layer.n_outputs();
                        irows = layer.n_orows();
                        icols = layer.n_ocols();
                }

                // create output layer
                conv_layer_t layer;
                n_params += layer.resize(idims, irows, icols, odims, irows, icols, "unit");
                layer.zero_params();
                m_layers.push_back(layer);

                if (n_params == 0)
                {
                        const string_t message = "convolution network: invalid convolution network model!";

                        log_error() << message;
                        throw std::runtime_error(message);
                }

                else
                {
                        print();
                }

                return n_params;
        }

        //-------------------------------------------------------------------------------------------------

        void conv_network_t::print() const
        {
                for (size_t l = 0; l < m_layers.size(); l ++)
                {
                        const conv_layer_t& layer = m_layers[l];

                        const auto message =
                                boost::format("convolution network: layer [%1%/%2%] %3% "\
                                              "(%4%x%5%x%6% * %7%x%8%x%9%x%10%) -> %11%x%12%x%13%.")
                                % (l + 1) % m_layers.size()
                                % layer.activation()
                                % layer.n_inputs() % layer.n_irows() % layer.n_icols()
                                % layer.kdata().n_dim1() % layer.kdata().n_dim2()
                                % layer.kdata().n_rows() % layer.kdata().n_cols()
                                % layer.n_outputs() % layer.n_orows() % layer.n_ocols();

                        log_info() << message.str();
                }
        }

        //-------------------------------------------------------------------------------------------------
}

