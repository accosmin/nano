#include "forward_network.h"
#include "layers/layer_convolution.h"
#include "core/logger.h"
#include "core/text.h"
#include "core/cast.h"
#include "core/tensor3d.h"

namespace ncv
{
        //-------------------------------------------------------------------------------------------------

        forward_network_t::forward_network_t(const string_t& params)
                :       m_params(params)
        {                
        }

        //-------------------------------------------------------------------------------------------------

        vector_t forward_network_t::value(const tensor3d_t& _input) const
        {
                const tensor3d_t* input = &_input;
                for (rlayers_t::const_iterator it = m_layers.begin(); it != m_layers.end(); ++ it)
                {
                        input = &(*it)->forward(*input);
                }

                const tensor3d_t& output = *input;
                vector_t result(output.size());
                serializer_t(result) << output;

                return result;
        }

        //-------------------------------------------------------------------------------------------------

        void forward_network_t::zero_grad() const
        {
                for (const rlayer_t& layer : m_layers)
                {
                        layer->zero_grad();
                }
        }

        //-------------------------------------------------------------------------------------------------

        void forward_network_t::cumulate_grad(const vector_t& vgradient) const
        {
                tensor3d_t _gradient(n_outputs(), 1, 1);
                deserializer_t(vgradient) >> _gradient;

                const tensor3d_t* gradient = &_gradient;
                for (rlayers_t::const_reverse_iterator it = m_layers.rbegin(); it != m_layers.rend(); ++ it)
                {
                        gradient = &(*it)->backward(*gradient);
                }
        }

        //-------------------------------------------------------------------------------------------------

        vector_t forward_network_t::grad() const
        {
                vector_t mgradient(n_parameters());

                serializer_t s(mgradient);
                for (const rlayer_t& layer : m_layers)
                {
                        layer->save_grad(s);
                }

                return mgradient;
        }

        //-------------------------------------------------------------------------------------------------

        bool forward_network_t::save_params(vector_t& x) const
        {
                if (math::cast<size_t>(x.size()) == n_parameters())
                {
                        serializer_t s(x);
                        for (const rlayer_t& layer : m_layers)
                        {
                                layer->save_params(s);
                        }

                        return true;
                }

                else
                {
                        return false;
                }
        }

        //-------------------------------------------------------------------------------------------------

        bool forward_network_t::load_params(const vector_t& x)
        {
                if (math::cast<size_t>(x.size()) == n_parameters())
                {
                        deserializer_t s(x);
                        for (const rlayer_t& layer : m_layers)
                        {
                                layer->load_params(s);
                        }

                        zero_grad();
                        return true;
                }

                else
                {
                        return false;
                }
        }

        //-------------------------------------------------------------------------------------------------

        void forward_network_t::zero_params()
        {
                for (const rlayer_t& layer : m_layers)
                {
                        layer->zero_params();
                }
        }

        //-------------------------------------------------------------------------------------------------

        void forward_network_t::random_params()
        {
                const scalar_t min = -1.0 / sqrt(n_parameters());
                const scalar_t max = +1.0 / sqrt(n_parameters());

                for (const rlayer_t& layer : m_layers)
                {
                        layer->random_params(min, max);
                }
        }

        //-------------------------------------------------------------------------------------------------

        bool forward_network_t::save(boost::archive::binary_oarchive& oa) const
        {
                // TODO

                return false;
        }

        //-------------------------------------------------------------------------------------------------

        bool forward_network_t::load(boost::archive::binary_iarchive& ia)
        {
                // TODO

                return false;
        }

        //-------------------------------------------------------------------------------------------------

        size_t forward_network_t::resize()
        {
                size_t irows = n_rows();
                size_t icols = n_cols();
                size_t idims = n_inputs();
                size_t n_params = 0;

                m_layers.clear();

                // create hidden layers
                strings_t net_params;
                text::split(net_params, m_params, text::is_any_of(";"));
                for (size_t l = 0; l < net_params.size(); l ++)
                {
                        strings_t layer_tokens;
                        text::split(layer_tokens, net_params[l], text::is_any_of(":"));
                        if (layer_tokens.size() != 2)
                        {
                                const string_t message = "invalid layer description <" +
                                        net_params[l] + ">! expecting <layer_id:layer_parameters>!";

                                log_error() << "forward network: " << message;
                                throw std::runtime_error(message);
                        }

                        const string_t layer_id = layer_tokens[0];
                        const string_t layer_params = layer_tokens[1];

                        const rlayer_t layer = layer_manager_t::instance().get(layer_id, layer_params);
                        if (!layer)
                        {
                                const string_t message = "invalid layer id <" + layer_id + ">!";

                                log_error() << "forward network: " << message;
                                throw std::runtime_error(message);
                        }

                        const size_t n_new_params = layer->resize(idims, irows, icols);
                        n_params += n_new_params;
                        m_layers.push_back(layer);

                        idims = layer->n_odims();
                        irows = layer->n_orows();
                        icols = layer->n_ocols();
                }

                // create output layer
                const rlayer_t layer(new conv_layer_t(
                        "convs=" + text::to_string(n_outputs()) + ","
                        "crows=" + text::to_string(irows) + ","
                        "ccols=" + text::to_string(icols)));
                n_params += layer->resize(idims, irows, icols);
                m_layers.push_back(layer);

                print();

                return n_params;
        }

        //-------------------------------------------------------------------------------------------------

        void forward_network_t::print() const
        {
                for (size_t l = 0; l < m_layers.size(); l ++)
                {
                        const rlayer_t& layer = m_layers[l];

                        log_info() << boost::format("feed-forward network: layer [%1%/%2%] %3%.")
                                % (l + 1) % m_layers.size() % layer->describe();
                }
        }

        //-------------------------------------------------------------------------------------------------
}

