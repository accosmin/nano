#include "forward_network.h"
#include "common/logger.h"
#include "common/math.hpp"
#include "tensor/util.hpp"

namespace ncv
{
        forward_network_t::forward_network_t(const string_t& parameters)
                :       model_t(parameters, "feed-forward network, parameters: [layer_id[:layer_parameters][;]]*")
        {                
        }

        forward_network_t::forward_network_t(const forward_network_t& other)
                :       model_t(other),
                        m_layers(other.m_layers)
        {
                for (size_t l = 0; l < n_layers(); l ++)
                {
                        m_layers[l] = other.m_layers[l]->clone();
                }
        }

        forward_network_t& forward_network_t::operator=(forward_network_t other)
        {
                if (this != &other)
                {
                        model_t::operator=(other);
                        std::swap(m_layers, other.m_layers);
                }

                return *this;
        }

        const tensor_t& forward_network_t::output(const tensor_t& _input) const
        {
                const tensor_t* input = &_input;
                for (rlayers_t::const_iterator it = m_layers.begin(); it != m_layers.end(); ++ it)
                {
                        const rlayer_t& layer = *it;

                        input = &layer->output(*input);
                }

		return *input;
        }

        const tensor_t& forward_network_t::igrad(const vector_t& _output) const
        {
                assert(static_cast<size_t>(_output.size()) == osize());
                assert(!m_layers.empty());

                // output (gradient)
                tensor_t output(osize(), 1, 1);
                tensor::load(output, _output.data());

                // backward step
                const tensor_t* poutput = &output;
                for (rlayers_t::const_reverse_iterator it = m_layers.rbegin(); it != m_layers.rend(); ++ it)
                {
                        const rlayer_t& layer = *it;

                        poutput = &layer->igrad(*poutput);
                }

                return *poutput;
        }

        vector_t forward_network_t::pgrad(const vector_t& _output) const
        {
                assert(static_cast<size_t>(_output.size()) == osize());
                assert(!m_layers.empty());

                // output (gradient)
                tensor_t output(osize(), 1, 1);
                tensor::load(output, _output.data());

                // parameter gradient
                vector_t gradient(psize());

                // backward step
                const tensor_t* poutput = &output;
                scalar_t* pgradient = gradient.data() + gradient.size();

                for (rlayers_t::const_reverse_iterator it = m_layers.rbegin(); it != m_layers.rend(); ++ it)
                {
                        const rlayer_t& layer = *it;

                        pgradient -= layer->psize();
                        layer->pgrad(*poutput, pgradient);

                        if ((++ it) != m_layers.rend())
                        {
                                poutput = &layer->igrad(*poutput);
                        }
                        -- it;
                }

                return gradient;
        }

        vector_t forward_network_t::params() const
        {
                vector_t x(psize());

                scalar_t* px = x.data() + x.size();
                for (rlayers_t::const_reverse_iterator it = m_layers.rbegin(); it != m_layers.rend(); ++ it)
                {
                        const rlayer_t& layer = *it;

                        px -= layer->psize();
                        layer->save_params(px);
                }
                
                return x;
        }

        bool forward_network_t::load_params(const vector_t& x)
        {
                if (math::cast<size_t>(x.size()) == psize())
                {
                        const scalar_t* px = x.data() + x.size();
                        for (rlayers_t::const_reverse_iterator it = m_layers.rbegin(); it != m_layers.rend(); ++ it)
                        {
                                const rlayer_t& layer = *it;

                                px -= layer->psize();
                                layer->load_params(px);
                        }

                        return true;
                }

                else
                {
                        return false;
                }
        }

        void forward_network_t::zero_params()
        {
                for (const rlayer_t& layer : m_layers)
                {
                        layer->zero_params();
                }
        }

        void forward_network_t::random_params()
        {
                for (const rlayer_t& layer : m_layers)
                {
                        const size_t fanin = layer->idims();
                        const size_t fanout = layer->odims();
                        const scalar_t min = -std::sqrt(6.0 / (1.0 + fanin + fanout));
                        const scalar_t max = +std::sqrt(6.0 / (1.0 + fanin + fanout));

                        layer->random_params(min, max);
                }
        }

        bool forward_network_t::save(boost::archive::binary_oarchive& oa) const
        {
                const vector_t p = this->params();

                oa << m_configuration;
                oa << p;

                return true;
        }

        bool forward_network_t::load(boost::archive::binary_iarchive& ia)
        {
                vector_t p;

                ia >> m_configuration;
                ia >> p;

                resize(true);
                load_params(p);

                return true;
        }

        size_t forward_network_t::resize(bool verbose)
        {
                tensor_t input(idims(), irows(), icols());
                size_t n_params = 0;

                m_layers.clear();

                strings_t layer_ids;

                // create layers
                strings_t net_params;
                text::split(net_params, configuration(), text::is_any_of(";"));
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

                        input.resize(layer->odims(), layer->orows(), layer->ocols());
                }

                // check output size to match the target
                if (    input.dims() != osize() ||
                        input.rows() != 1 ||
                        input.cols() != 1)
                {
                        const string_t message = "miss-matching output size! expecting " +
                                text::to_string(osize()) + "!";

                        log_error() << "forward network: " << message;
                        throw std::runtime_error(message);
                }

                if (verbose)
                {
                        print(layer_ids);
                }

                return n_params;
        }

        void forward_network_t::print(const strings_t& layer_ids) const
        {
                assert(n_layers() == layer_ids.size());

                for (size_t l = 0; l < n_layers(); l ++)
                {
                        const rlayer_t& layer = m_layers[l];

                        log_info() <<
                                boost::format("forward network [%1%/%2%]: [%3%] (%4%x%5%x%6%) -> (%7%x%8%x%9%).")
                                % (l + 1) % m_layers.size() % layer_ids[l]
                                % layer->idims() % layer->irows() % layer->icols()
                                % layer->odims() % layer->orows() % layer->ocols();
                }
        }

        size_t forward_network_t::psize() const
        {
                size_t nparams = 0;
                for (const rlayer_t& layer : m_layers)
                {
                        nparams += layer->psize();
                }

                return nparams;
        }
}

