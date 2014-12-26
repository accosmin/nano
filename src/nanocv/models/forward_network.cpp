#include "forward_network.h"
#include "io/logger.h"
#include "common/math.hpp"
#include "common/cast.hpp"
#include "tensor/serialize.hpp"
#include <iomanip>

namespace ncv
{
        forward_network_t::forward_network_t(const string_t& parameters)
                :       model_t(parameters)
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

        const tensor_t& forward_network_t::ginput(const vector_t& _output) const
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

                        poutput = &layer->ginput(*poutput);
                }

                return *poutput;
        }

        vector_t forward_network_t::gparam(const vector_t& _output) const
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
                scalar_t* gparamient = gradient.data() + gradient.size();

                for (rlayers_t::const_reverse_iterator it = m_layers.rbegin(); it != m_layers.rend(); ++ it)
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

                return gradient;
        }

        bool forward_network_t::save_params(vector_t& x) const
        {
                const size_t psize = this->psize();
                if (psize != static_cast<size_t>(x.size()))
                {
                        x.resize(psize);
                }

                scalar_t* px = x.data() + x.size();
                for (rlayers_t::const_reverse_iterator it = m_layers.rbegin(); it != m_layers.rend(); ++ it)
                {
                        const rlayer_t& layer = *it;

                        px -= layer->psize();
                        layer->save_params(px);
                }

                return true;
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
                oa << m_configuration;

                for (rlayers_t::const_iterator it = m_layers.begin(); it != m_layers.end(); ++ it)
                {
                        (*it)->save(oa);
                }

                return true;
        }

        bool forward_network_t::load(boost::archive::binary_iarchive& ia)
        {
                ia >> m_configuration;
                resize(true);

                for (rlayers_t::const_iterator it = m_layers.begin(); it != m_layers.end(); ++ it)
                {
                        (*it)->load(ia);
                }

                return true;
        }

        size_t forward_network_t::resize(bool verbose)
        {
                tensor_t input(idims(), irows(), icols());
                size_t n_params = 0;

                m_layers.clear();

                strings_t layer_ids;

                // create layers
                const string_t config = this->configuration();

                strings_t net_params;
                text::split(net_params, config, text::is_any_of(";"));
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

                const scalar_t mflops_inv = 1.0 / (1024.0 * 1024.0);

                scalar_t model_output_mflops = 0.0;
                scalar_t model_ginput_mflops = 0.0;
                scalar_t model_gparam_mflops = 0.0;

                for (size_t l = 0; l < n_layers(); l ++)
                {
                        const rlayer_t& layer = m_layers[l];

                        log_info() << "forward network [" << (l + 1) << "/" << m_layers.size() << "]: "
                                   << "[" << layer_ids[l] << "] "
                                   << "(" << layer->idims() << "x" << layer->irows() << "x" << layer->icols() << ") -> "
                                   << "(" << layer->odims() << "x" << layer->orows() << "x" << layer->ocols() << ").";

                        const scalar_t output_mflops = mflops_inv * static_cast<scalar_t>(layer->output_flops());
                        const scalar_t ginput_mflops = mflops_inv * static_cast<scalar_t>(layer->ginput_flops());
                        const scalar_t gparam_mflops = mflops_inv * static_cast<scalar_t>(layer->gparam_flops());

                        model_output_mflops += output_mflops;
                        model_ginput_mflops += ginput_mflops;
                        model_gparam_mflops += gparam_mflops;
                }

                log_info() << "forward network [MFLOPs]"
                           << ": output = " << std::setprecision(3) << model_output_mflops
                           << ", ginput = " << std::setprecision(3) << model_ginput_mflops
                           << ", gparam = " << std::setprecision(3) << model_gparam_mflops;
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

