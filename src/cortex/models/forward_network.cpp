#include "math/cast.hpp"
#include "text/align.hpp"
#include "cortex/logger.h"
#include "text/algorithm.h"
#include "math/numeric.hpp"
#include "forward_network.h"
#include "text/to_string.hpp"
#include "tensor/serialize.hpp"
#include <iomanip>

namespace nano
{
        forward_network_t::forward_network_t(const string_t& parameters) :
                model_t(parameters)
        {
        }

        forward_network_t::forward_network_t(const forward_network_t& other) :
                model_t(other),
                m_layers(other.m_layers),
                m_gparam(other.m_gparam),
                m_odata(other.m_odata)
        {
                for (size_t l = 0; l < n_layers(); ++ l)
                {
                        m_layers[l] = other.m_layers[l]->clone();
                }
        }

        const tensor3d_t& forward_network_t::output(const tensor3d_t& _input)
        {
                const tensor3d_t* input = &_input;
                for (auto it = m_layers.begin(); it != m_layers.end(); ++ it)
                {
                        const rlayer_t& layer = *it;

                        input = &layer->output(*input);
                }

                return *input;
        }

        const tensor3d_t& forward_network_t::ginput(const vector_t& _output)
        {
                assert(_output.size() == osize());
                assert(!m_layers.empty());

                // output (gradient)
                m_odata = tensor::map_tensor(_output.data(), osize(), tensor_size_t(1), tensor_size_t(1));

                // backward step
                const tensor3d_t* poutput = &m_odata;
                for (auto it = m_layers.rbegin(); it != m_layers.rend(); ++ it)
                {
                        const rlayer_t& layer = *it;

                        poutput = &layer->ginput(*poutput);
                }

                return *poutput;
        }

        const vector_t& forward_network_t::gparam(const vector_t& _output)
        {
                assert(_output.size() == osize());
                assert(!m_layers.empty());

                // output (gradient)
                m_odata = tensor::map_tensor(_output.data(), osize(), tensor_size_t(1), tensor_size_t(1));

                // parameter gradient
                m_gparam.resize(psize());

                // backward step
                const tensor3d_t* poutput = &m_odata;
                scalar_t* gparamient = m_gparam.data() + m_gparam.size();

                for (auto it = m_layers.rbegin(); it != m_layers.rend(); ++ it)
                {
                        const rlayer_t& layer = *it;

                        gparamient -= layer->psize();
                        layer->gparam(*poutput, gparamient);

                        ++ it;
                        if (it != m_layers.rend())
                        {
                                poutput = &layer->ginput(*poutput);
                        }
                        -- it;
                }

                return m_gparam;
        }

        bool forward_network_t::save_params(vector_t& x) const
        {
                const auto psize = this->psize();
                if (psize != x.size())
                {
                        x.resize(psize);
                }

                scalar_t* px = x.data() + x.size();
                for (auto it = m_layers.rbegin(); it != m_layers.rend(); ++ it)
                {
                        const rlayer_t& layer = *it;

                        px -= layer->psize();
                        layer->save_params(px);
                }

                return true;
        }

        bool forward_network_t::load_params(const vector_t& x)
        {
                if (x.size() == psize())
                {
                        const scalar_t* px = x.data() + x.size();
                        for (auto it = m_layers.rbegin(); it != m_layers.rend(); ++ it)
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
                        const auto fanin = layer->idims() * layer->irows() * layer->icols();
                        const auto fanout = layer->odims() * layer->orows() * layer->ocols();

                        const auto div = static_cast<scalar_t>(fanin + fanout);
                        const auto min = -std::sqrt(6.0 / (1.0 + div));
                        const auto max = +std::sqrt(6.0 / (1.0 + div));

                        layer->random_params(min, max);
                }
        }

        tensor_size_t forward_network_t::resize(bool verbose)
        {
                tensor3d_t input(idims(), irows(), icols());
                tensor_size_t n_params = 0;

                m_layers.clear();

                strings_t layer_ids;

                // create layers
                const string_t config = this->configuration();

                const strings_t net_params = nano::split(config, ";");
                for (size_t l = 0; l < net_params.size(); ++ l)
                {
                        if (net_params[l].empty())
                        {
                                continue;
                        }

                        const strings_t layer_tokens = nano::split(net_params[l], ":");
                        if (layer_tokens.size() != 2 && layer_tokens.size() != 1)
                        {
                                log_error() << "forward network: invalid layer description <"
                                            << net_params[l] << ">! expecting <layer_id[:layer_parameters]>!";
                                throw std::invalid_argument("invalid layer description");
                        }

                        const string_t layer_id = layer_tokens[0];
                        const string_t layer_params = layer_tokens.size() == 2 ? layer_tokens[1] : string_t();

                        const rlayer_t layer = nano::get_layers().get(layer_id, layer_params);

                        n_params += layer->resize(input);
                        m_layers.push_back(layer);
                        layer_ids.push_back(layer_id);

                        input.resize(layer->odims(), layer->orows(), layer->ocols());
                }

                // check output size to match the target
                if (    input.size<0>() != osize() ||
                        input.size<1>() != 1 ||
                        input.size<2>() != 1)
                {
                        log_error() << "forward network: miss-matching output size! expecting " << osize() << "!";
                        throw std::invalid_argument("invalid output layer description");
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

                for (size_t l = 0; l < n_layers(); ++ l)
                {
                        const rlayer_t& layer = m_layers[l];

                        log_info()
                                << "forward network [" << align(to_string(l + 1), 2, alignment::right, '0')
                                << "/" << align(to_string(m_layers.size()), 2, alignment::right, '0') << "]: "
                                << "[" << align(layer_ids[l], 12, alignment::right, '.') << "] "
                                << "in(" << layer->idims() << "x" << layer->irows() << "x" << layer->icols() << ") -> "
                                << "out(" << layer->odims() << "x" << layer->orows() << "x" << layer->ocols() << ") == "
                                << layer->psize() << " parameters.";
                }
        }

        tensor_size_t forward_network_t::psize() const
        {
                tensor_size_t nparams = 0;
                for (const rlayer_t& layer : m_layers)
                {
                        nparams += layer->psize();
                }

                return nparams;
        }
}

