#include "model_conv_network.h"
#include "core/logger.h"
#include "core/thread.h"
#include "core/optimize.h"
#include "core/timer.h"

namespace ncv
{
        //-------------------------------------------------------------------------------------------------

        conv_network_model_t::conv_network_model_t(const string_t& params)
                :       model_t("convolution network",
                                "parameters: color=luma[luma,rgba],network=[16:8x8[,...]]")
        {
                m_color_param = text::from_params<color_mode>(params, "color", ncv::color_mode::luma);

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
                                strings_t layer_tokens;
                                text::split(layer_tokens, network_tokens[l], text::is_any_of(":x"));
                                if (layer_tokens.size() != 3)
                                {
                                        throw std::runtime_error("invalid layer description <" +
                                                                 network_tokens[l] + "> for the nework <" +
                                                                 network_desc + ">!");
                                }

                                // convolutions ...
                                const size_t convs = text::from_string<size_t>(layer_tokens[0]);
                                const size_t crows = text::from_string<size_t>(layer_tokens[1]);
                                const size_t ccols = text::from_string<size_t>(layer_tokens[2]);

                                m_network_param.push_back(convs);
                                m_network_param.push_back(crows);
                                m_network_param.push_back(ccols);
                        }
                }
        }

        //-------------------------------------------------------------------------------------------------

        tensor3d_t conv_network_model_t::make_input(const image_t& image, coord_t x, coord_t y) const
        {
                tensor3d_t data;

                const rect_t region = geom::make_rect(x, y, n_cols(), n_rows());
                switch (m_color_param)
                {
                case color_mode::luma:
                        data.resize(1, n_rows(), n_cols());
                        data.data(0) = image.make_luma(region);
                        break;

                case color_mode::rgba:
                        data.resize(3, n_rows(), n_cols());
                        data.data(0) = image.make_red(region);
                        data.data(1) = image.make_green(region);
                        data.data(2) = image.make_blue(region);
                        break;
                }

                return data;
        }

        //-------------------------------------------------------------------------------------------------

        tensor3d_t conv_network_model_t::make_input(const image_t& image, const rect_t& region) const
        {
                return make_input(image, geom::left(region), geom::top(region));
        }

        //-------------------------------------------------------------------------------------------------

        size_t conv_network_model_t::n_inputs() const
        {
                switch (m_color_param)
                {
                case color_mode::rgba:
                        return 3;

                case color_mode::luma:
                default:
                        return 1;
                }
        }

        //-------------------------------------------------------------------------------------------------

        vector_t conv_network_model_t::forward(const image_t& image, coord_t x, coord_t y) const
        {
                tensor3d_t input = make_input(image, x, y);

                for (size_t l = 0; l < m_layers.size(); l ++)
                {
                        const tensor4d_t& layer = m_layers[l];
                        input = layer.forward(input);
                }

                return input.to_vector();
        }

        //-------------------------------------------------------------------------------------------------

        bool conv_network_model_t::save(boost::archive::binary_oarchive& oa) const
        {
                oa << m_layers;
                oa << m_color_param;
                oa << m_network_param;

                return true;    // fixme: how to check status of the stream?!
        }

        //-------------------------------------------------------------------------------------------------

        bool conv_network_model_t::load(boost::archive::binary_iarchive& ia)
        {
                ia >> m_layers;
                ia >> m_color_param;
                ia >> m_network_param;

                return true;    // fixme: how to check status of the stream?!
        }

        //-------------------------------------------------------------------------------------------------

        size_t conv_network_model_t::resize()
        {
                size_t n_params = 0;

                size_t idims = n_inputs();
                size_t irows = n_rows();
                size_t icols = n_cols();

                const size_t n_layers = m_network_param.size() / 3;
                assert(m_network_param.size() % 3 == 0);

                // create hidden layers
                m_layers.resize(n_layers + 1);
                for (size_t l = 0; l < n_layers; l ++)
                {
                        const size_t convs = m_network_param[3 * l + 0];
                        const size_t crows = m_network_param[3 * l + 1];
                        const size_t ccols = m_network_param[3 * l + 2];

                        if (convs < 1 ||
                            crows < 1 || crows >= irows ||
                            ccols < 1 || ccols >= icols)
                        {
                                throw std::runtime_error("invalid network description for the inputs " +
                                                         text::to_string(n_inputs()) + "x" +
                                                         text::to_string(n_rows()) + "x" +
                                                         text::to_string(n_cols()) + "!");
                        }

                        n_params += m_layers[l].resize(idims, convs, crows, ccols);

                        idims = convs;
                        irows -= crows;
                        icols -= ccols;
                }

                // create output layer
                n_params += m_layers[n_layers].resize(idims, n_outputs(), irows, icols);

                return n_params;
        }

        //-------------------------------------------------------------------------------------------------

        void conv_network_model_t::zero()
        {
                for (tensor4d_t& layer : m_layers)
                {
                        layer.zero();
                }
        }

        //-------------------------------------------------------------------------------------------------

        void conv_network_model_t::random()
        {
                const scalar_t min = -1.0 / sqrt(n_parameters());
                const scalar_t max = +1.0 / sqrt(n_parameters());

                for (tensor4d_t& layer : m_layers)
                {
                        layer.random(min, max);
                }
        }

        //-------------------------------------------------------------------------------------------------

        vector_t conv_network_model_t::serialize() const
        {
                vector_t params(n_parameters());

                serializer_t s(params);
                for (const tensor4d_t& layer : m_layers)
                {
                        layer.serialize(s);
                }

                return params;
        }

        //-------------------------------------------------------------------------------------------------

        void conv_network_model_t::deserialize(const vector_t& params)
        {
                deserializer_t s(params);
                for (tensor4d_t& layer : m_layers)
                {
                        layer.deserialize(s);
                }
        }

        //-------------------------------------------------------------------------------------------------

        void conv_network_model_t::cum_loss(const task_t& task, const loss_t& loss, const sample_t& sample,
                conv_network_t& network) const
        {
                const image_t& image = task.image(sample.m_index);
                const vector_t target = image.make_target(sample.m_region);
                if (image.has_target(target))
                {
//                        const matrices_t input = make_input(image, sample.m_region);
//                        data.forward(input, target, loss);
                }
        }

        //-------------------------------------------------------------------------------------------------

        void conv_network_model_t::cum_grad(const task_t& task, const loss_t& loss, const sample_t& sample,
                conv_network_t& network) const
        {
                const image_t& image = task.image(sample.m_index);
                const vector_t target = image.make_target(sample.m_region);
                if (image.has_target(target))
                {
//                        const matrices_t input = make_input(image, sample.m_region);
//                        data.backward(input, target, loss);
                }
        }

        //-------------------------------------------------------------------------------------------------

        static void update(const optimize::result_t& result, timer_t& timer)
        {
                ncv::log_info() << "convolution model: state [loss = " << result.optimum().f
                                << ", gradient = " << result.optimum().g.lpNorm<Eigen::Infinity>()
                                << "] updated in " << timer.elapsed() << ".";
                timer.start();
        }

        //-------------------------------------------------------------------------------------------------

        bool conv_network_model_t::train(const task_t& task, const samples_t& samples, const loss_t& loss)
        {
                return false;

//                // optimization problem: size
//                auto opt_fn_size = [&] ()
//                {
//                        return n_parameters();
//                };

//                // optimization problem: function value
//                auto opt_fn_fval = [&] (const vector_t& x)
//                {
//                        olayer_t cum_data(n_outputs(), n_inputs(), n_rows(), n_cols());

//                        thread_loop_cumulate<olayer_t>
//                        (
//                                samples.size(),
//                                [&] (olayer_t& data)
//                                {
//                                        data.resize(n_outputs(), n_inputs(), n_rows(), n_cols());
//                                        deserializer_t s(x);
//                                        data.deserialize(s);
//                                },
//                                [&] (size_t i, olayer_t& data)
//                                {
//                                        cum_loss(task, loss, samples[i], data);
//                                },
//                                [&] (const olayer_t& data)
//                                {
//                                        cum_data += data;
//                                }
//                        );

//                        return cum_data.loss() / cum_data.count();
//                };

//                // optimization problem: function value & gradient
//                auto opt_fn_fval_grad = [&] (const vector_t& x, vector_t& gx)
//                {
//                        olayer_t cum_data(n_outputs(), n_inputs(), n_rows(), n_cols());

//                        thread_loop_cumulate<olayer_t>
//                        (
//                                samples.size(),
//                                [&] (olayer_t& data)
//                                {
//                                        data.resize(n_outputs(), n_inputs(), n_rows(), n_cols());
//                                        deserializer_t s(x);
//                                        data.deserialize(s);
//                                },
//                                [&] (size_t i, olayer_t& data)
//                                {
//                                        cum_grad(task, loss, samples[i], data);
//                                },
//                                [&] (const olayer_t& data)
//                                {
//                                        cum_data += data;
//                                }
//                        );

//                        gx.resize(n_parameters());
//                        serializer_t s(gx);
//                        cum_data.gserialize(s);
//                        gx /= cum_data.count();

//                        return cum_data.loss() / cum_data.count();
//                };

//                const optimize::problem_t problem(opt_fn_size, opt_fn_fval, opt_fn_fval_grad);

//                // optimize
//                static const size_t opt_iters = 256;
//                static const scalar_t opt_eps = 1e-5;
//                static const size_t opt_history = 8;

//                timer_t timer;
//                const optimize::result_t res = optimize::lbfgs(
//                        problem, serialize(),
//                        opt_iters, opt_eps, opt_history,
//                        std::bind(update, _1, std::ref(timer)));

//                deserialize(res.optimum().x);

//                // OK
//                log_info() << "linear model: optimum [loss = " << res.optimum().f
//                           << ", gradient = " << res.optimum().g.norm() << "]"
//                           << ", iterations = [" << res.iterations() << "/" << opt_iters
//                           << "], speed = [" << res.speed().avg() << " +/- " << res.speed().stdev() << "].";

//                return true;
        }

        //-------------------------------------------------------------------------------------------------
}

