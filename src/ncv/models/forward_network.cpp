#include "forward_network.h"
#include "common/logger.h"
#include "common/math.hpp"
#include "text.h"

namespace ncv
{
        /////////////////////////////////////////////////////////////////////////////////////////

        forward_network_t::forward_network_t(const string_t& params)
                :       m_params(params)
        {                
        }

        /////////////////////////////////////////////////////////////////////////////////////////

        vector_t forward_network_t::value(const tensor_t& _input) const
        {
                const tensor_t* input = &_input;
                for (rlayers_t::const_iterator it = m_layers.begin(); it != m_layers.end(); ++ it)
                {
                        input = &(*it)->forward(*input);
                }

                const tensor_t& output = *input;
                return output.vector();
        }

        /////////////////////////////////////////////////////////////////////////////////////////

        vector_t forward_network_t::gradient(const vector_t& ograd) const
        {
                assert(static_cast<size_t>(ograd.size()) == n_outputs());

                tensor_t _gradient(n_outputs(), 1, 1);
                ivectorizer_t(ograd) >> _gradient;

                const tensor_t* gradient = &_gradient;
                for (rlayers_t::const_reverse_iterator it = m_layers.rbegin(); it != m_layers.rend(); ++ it)
                {
                        gradient = &(*it)->backward(*gradient);
                }

                vector_t mgradient(n_parameters());

                ovectorizer_t s(mgradient);
                for (const rlayer_t& layer : m_layers)
                {
                        layer->save_grad(s);
                }

                return mgradient;
        }

        /////////////////////////////////////////////////////////////////////////////////////////

        vector_t forward_network_t::params() const
        {
                vector_t x(n_parameters());

                ovectorizer_t s(x);
                for (const rlayer_t& layer : m_layers)
                {
                        layer->save_params(s);
                }

                return x;
        }

        /////////////////////////////////////////////////////////////////////////////////////////

        bool forward_network_t::load_params(const vector_t& x)
        {
                if (math::cast<size_t>(x.size()) == n_parameters())
                {
                        ivectorizer_t s(x);
                        for (const rlayer_t& layer : m_layers)
                        {
                                layer->load_params(s);
                        }

                        return true;
                }

                else
                {
                        return false;
                }
        }

        /////////////////////////////////////////////////////////////////////////////////////////

        void forward_network_t::zero_params()
        {
                for (const rlayer_t& layer : m_layers)
                {
                        layer->zero_params();
                }
        }

        /////////////////////////////////////////////////////////////////////////////////////////

        void forward_network_t::random_params()
        {
                for (const rlayer_t& layer : m_layers)
                {
                        const size_t fanin = layer->input().dims();
                        const size_t fanout = layer->output().dims();
                        const scalar_t min = -std::sqrt(6.0 / (1.0 + fanin + fanout));
                        const scalar_t max = +std::sqrt(6.0 / (1.0 + fanin + fanout));

                        layer->random_params(min, max);
                }
        }

        /////////////////////////////////////////////////////////////////////////////////////////

        bool forward_network_t::save(boost::archive::binary_oarchive& oa) const
        {
                const vector_t p = this->params();

                oa << m_params;
                oa << p;

                return true;
        }

        /////////////////////////////////////////////////////////////////////////////////////////

        bool forward_network_t::load(boost::archive::binary_iarchive& ia)
        {
                vector_t p;

                ia >> m_params;
                ia >> p;

                resize();
                load_params(p);

                return true;
        }

        /////////////////////////////////////////////////////////////////////////////////////////

        size_t forward_network_t::resize()
        {
                tensor_t input(n_inputs(), n_rows(), n_cols());
                size_t n_params = 0;

                m_layers.clear();
                strings_t layer_ids;

                // create hidden layers
                strings_t net_params;
                text::split(net_params, m_params, text::is_any_of(";"));
                for (size_t l = 0; l < net_params.size(); l ++)
                {
                        if (net_params[l].empty())
                        {
                                continue;
                        }

                        strings_t layer_tokens;
                        text::split(layer_tokens, net_params[l], text::is_any_of(":"));
                        if (layer_tokens.size() != 2 && layer_tokens.size() != 1)
                        {
                                const string_t message = "invalid layer description <" +
                                        net_params[l] + ">! expecting <layer_id[:layer_parameters]>!";

                                log_error() << "forward network: " << message;
                                throw std::runtime_error(message);
                        }

                        const string_t layer_id = layer_tokens[0];
                        const string_t layer_params = layer_tokens.size() == 2 ? layer_tokens[1] : string_t();

                        const rlayer_t layer = layer_manager_t::instance().get(layer_id, layer_params);
                        if (!layer)
                        {
                                const string_t message = "invalid layer id <" + layer_id + ">!";

                                log_error() << "forward network: " << message;
                                throw std::runtime_error(message);
                        }

                        n_params += layer->resize(input);
                        m_layers.push_back(layer);
                        layer_ids.push_back(layer_id);

                        input = layer->output();
                }

                // Create the output layer
                const string_t layer_id = "linear";
                const rlayer_t layer = layer_manager_t::instance().get(layer_id, "dims=" + text::to_string(n_outputs()));
                n_params += layer->resize(input);
                m_layers.push_back(layer);
                layer_ids.push_back(layer_id);

                print(layer_ids);

                return n_params;
        }

        /////////////////////////////////////////////////////////////////////////////////////////

        void forward_network_t::print(const strings_t& layer_ids) const
        {
                for (size_t l = 0; l < m_layers.size(); l ++)
                {
                        const rlayer_t& layer = m_layers[l];

                        log_info() <<
                                boost::format("feed-forward network [%1%/%2%]: [%3%] (%4%x%5%x%6%) -> (%7%x%8%x%9%).")
                                % (l + 1) % m_layers.size() % layer_ids[l]
                                % layer->input().dims() % layer->input().rows() % layer->input().cols()
                                % layer->output().dims() % layer->output().rows() % layer->output().cols();
                }
        }

        /////////////////////////////////////////////////////////////////////////////////////////

        forward_network_t::robject_t forward_network_t::clone() const
        {
                return forward_network_t::robject_t(new forward_network_t(*this));
        }

        /////////////////////////////////////////////////////////////////////////////////////////

        forward_network_t::robject_t forward_network_t::clone(const std::string& params) const
        {
                forward_network_t* result = new forward_network_t(params);

                result->m_layers.resize(m_layers.size());
                for (size_t l = 0; l < m_layers.size(); l ++)
                {
                        result->m_layers[l] = m_layers[l]->clone();
                }

                return forward_network_t::robject_t(result);
        }

        /////////////////////////////////////////////////////////////////////////////////////////
}

