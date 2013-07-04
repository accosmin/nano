#include "model_conv_network.h"
#include "core/logger.h"
#include "core/timer.h"
#include "activation/activation.h"

namespace ncv
{
        //-------------------------------------------------------------------------------------------------

        conv_network_model_t::conv_network_model_t(const string_t& params)
                :       model_t("parameters: network=[16:8:8:activation]*")
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

        vector_t conv_network_model_t::process(const tensor3d_t& input) const
        {
                const tensor3d_t& output = forward(input);

                vector_t result(output.size());
                serializer_t(result) << output;

                return result;
        }

        //-------------------------------------------------------------------------------------------------

        bool conv_network_model_t::save(boost::archive::binary_oarchive& oa) const
        {
                oa << m_layers;
                oa << m_params;

                return true;    // fixme: how to check status of the stream?!
        }

        //-------------------------------------------------------------------------------------------------

        bool conv_network_model_t::load(boost::archive::binary_iarchive& ia)
        {
                ia >> m_layers;
                ia >> m_params;

                return true;    // fixme: how to check status of the stream?!
        }

        //-------------------------------------------------------------------------------------------------

        bool conv_network_model_t::save(vector_t& x) const
        {
                if (static_cast<size_t>(x.size()) == n_parameters())
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

        bool conv_network_model_t::load(const vector_t& x)
        {
                if (static_cast<size_t>(x.size()) == n_parameters())
                {
                        deserializer_t(x) >> m_layers;
                        return true;
                }

                else
                {
                        return false;
                }
        }

        //-------------------------------------------------------------------------------------------------

        size_t conv_network_model_t::resize()
        {
                const size_t n_params = conv_layer_t::make_network(
                        n_inputs(), n_rows(), n_cols(), m_params, n_outputs(),
                        m_layers);

                if (n_params == 0)
                {
                        throw std::runtime_error("invalid convolution network model!");
                }

                else
                {
                        conv_layer_t::print_network(m_layers);
                }

                return n_params;
        }

        //-------------------------------------------------------------------------------------------------

        void conv_network_model_t::zero()
        {
                for (conv_layer_t& layer : m_layers)
                {
                        layer.zero();
                }
        }

        //-------------------------------------------------------------------------------------------------

        void conv_network_model_t::random()
        {
                const scalar_t min = -1.0 / sqrt(n_parameters());
                const scalar_t max = +1.0 / sqrt(n_parameters());

                for (conv_layer_t& layer : m_layers)
                {
                        layer.random(min, max);
                }
        }

        //-------------------------------------------------------------------------------------------------

        void conv_network_model_t::prune(data_t& data) const
        {
                data.m_indices.clear();

                for (size_t i = 0; i < data.m_samples.size(); i ++)
                {
                        const sample_t& sample = data.m_samples[i];

                        const image_t& image = data.m_task.image(sample.m_index);
                        const vector_t target = image.make_target(sample.m_region);
                        if (image.has_target(target))
                        {
                                data.m_indices.push_back(i);
                        }
                }
        }

        //-------------------------------------------------------------------------------------------------

        scalar_t conv_network_model_t::value(const data_t& data, const loss_t& loss) const
        {
                scalar_t lvalue = 0.0;
                size_t lcount = 0;

                timer_t timer;

                for (size_t i = 0; i < data.m_indices.size(); i ++)
                {
                        const sample_t& sample = data.m_samples[i];

                        const image_t& image = data.m_task.image(sample.m_index);
                        const vector_t target = image.make_target(sample.m_region);

                        // forward: network output
                        const tensor3d_t input = make_input(image, sample.m_region);
                        const tensor3d_t& output = forward(input);

                        // loss value
                        vector_t voutput(n_outputs());
                        serializer_t(voutput) << output;

                        lvalue += loss.value(target, voutput);
                        lcount ++;
                }

                lvalue /= (lcount == 0) ? 1.0 : lcount;

                std::cout << "::value: count = " << lcount
                          << ", loss = " << lvalue
                          << " done in " << timer.elapsed() << std::endl;

                return lvalue;
        }

        //-------------------------------------------------------------------------------------------------

        scalar_t conv_network_model_t::vgrad(const data_t& data, const loss_t& loss, vector_t& grad) const
        {
                scalar_t lvalue = 0.0;
                size_t lcount = 0;
                for (const conv_layer_t& layer : m_layers)
                {
                        layer.zero_grad();
                }

                const timer_t timer;

                for (size_t i = 0; i < data.m_indices.size(); i ++)
                {
                        const sample_t& sample = data.m_samples[i];

                        const image_t& image = data.m_task.image(sample.m_index);
                        const vector_t target = image.make_target(sample.m_region);

                        // forward: network output
                        const tensor3d_t input = make_input(image, sample.m_region);
                        const tensor3d_t& output = forward(input);

                        // loss value & gradient
                        vector_t voutput(n_outputs());
                        serializer_t(voutput) << output;

                        lvalue += loss.value(target, voutput);
                        lcount ++;

                        const vector_t vgradient = loss.vgrad(target, voutput);

                        // backward: network gradient
                        tensor3d_t gradient(n_outputs(), 1, 1);
                        deserializer_t(vgradient) >> gradient;

                        backward(gradient);
                }

                grad.resize(n_parameters());

                serializer_t s(grad);
                for (const conv_layer_t& layer : m_layers)
                {
                        s << layer.gdata();
                }

                grad /= (lcount == 0) ? 1.0 : lcount;
                lvalue /= (lcount == 0) ? 1.0 : lcount;

                std::cout << "::vgrad: count = " << lcount
                          << ", loss = " << lvalue
                          << ", grad = [" << grad.minCoeff() << ", " << grad.maxCoeff()
                          << "] done in " << timer.elapsed() << std::endl;

                return lvalue;
        }

        //-------------------------------------------------------------------------------------------------

        const tensor3d_t& conv_network_model_t::forward(const tensor3d_t& _input) const
        {
                const tensor3d_t* input = &_input;
                for (conv_layers_t::const_iterator it = m_layers.begin(); it != m_layers.end(); ++ it)
                {
                        input = &it->forward(*input);
                }

                return *input;
        }

        //-------------------------------------------------------------------------------------------------

        void conv_network_model_t::backward(const tensor3d_t& _gradient) const
        {
                const tensor3d_t* gradient = &_gradient;
                for (conv_layers_t::const_reverse_iterator it = m_layers.rbegin(); it != m_layers.rend(); ++ it)
                {
                        gradient = &it->backward(*gradient);
                }
        }

        //-------------------------------------------------------------------------------------------------
}

