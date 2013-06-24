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
                        data(0) = image.make_luma(region);
                        break;

                case color_mode::rgba:
                        data.resize(3, n_rows(), n_cols());
                        data(0) = image.make_red(region);
                        data(1) = image.make_green(region);
                        data(2) = image.make_blue(region);
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
                const tensor3d_t input = make_input(image, x, y);
                const tensor3d_t& output = forward(input);

                vector_t result(output.size());
                serializer_t(result) << output;

                return result;
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
                m_layers.clear();
                for (size_t l = 0; l < n_layers; l ++)
                {
                        conv_layer_t layer;
                        const size_t convs = m_network_param[3 * l + 0];
                        const size_t crows = m_network_param[3 * l + 1];
                        const size_t ccols = m_network_param[3 * l + 2];

                        const size_t n_new_params = layer.resize(idims, irows, icols, convs, crows, ccols);
                        if (n_new_params < 1)
                        {
                                throw std::runtime_error("invalid network description for the inputs " +
                                                         text::to_string(n_inputs()) + "x" +
                                                         text::to_string(n_rows()) + "x" +
                                                         text::to_string(n_cols()) + "!");
                        }
                        n_params += n_new_params;
                        m_layers.push_back(layer);

                        idims = layer.n_outputs();
                        irows = layer.n_orows();
                        icols = layer.n_ocols();
                }

                // create output layer
                conv_layer_t layer;
                n_params += layer.resize(idims, irows, icols, n_outputs(), irows, icols);
                m_layers.push_back(layer);

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

        const tensor3d_t& conv_network_model_t::forward(const tensor3d_t& _input) const
        {
                const tensor3d_t* input = &_input;
                for (layers_t::const_iterator it = m_layers.begin(); it != m_layers.end(); ++ it)
                {
                        input = &it->forward(*input);
                }

                return *input;
        }

        //-------------------------------------------------------------------------------------------------

        void conv_network_model_t::backward(const tensor3d_t& _gradient)
        {
                const tensor3d_t* gradient = &_gradient;
                for (layers_t::reverse_iterator it = m_layers.rbegin(); it != m_layers.rend(); ++ it)
                {
                        gradient = &it->backward(*gradient);
                }
        }

        //-------------------------------------------------------------------------------------------------

        scalar_t conv_network_model_t::value(const task_t& task, const loss_t& loss, const samples_t& samples)
        {
                scalar_t lvalue = 0.0;
                size_t lcount = 0;

                for (size_t i = 0; i < samples.size(); i ++)
                {
                        const sample_t& sample = samples[i];

                        const image_t& image = task.image(sample.m_index);
                        const vector_t target = image.make_target(sample.m_region);
                        if (image.has_target(target))
                        {
                                // forward: network output
                                const tensor3d_t input = make_input(image, sample.m_region);
                                const tensor3d_t& output = forward(input);

                                // loss value
                                vector_t voutput(n_outputs());
                                serializer_t(voutput) << output;

                                lvalue += loss.value(target, voutput);
                                lcount ++;
                        }
                }

                return lcount == 0 ? 0.0 : lvalue / lcount;
        }

        //-------------------------------------------------------------------------------------------------

        scalar_t conv_network_model_t::vgrad(const task_t& task, const loss_t& loss, const samples_t& samples)
        {
                scalar_t lvalue = 0.0;
                size_t lcount = 0;

                std::cout << "vgrad - begin " << std::endl;

                for (size_t i = 0; i < samples.size(); i ++)
                {
                        const sample_t& sample = samples[i];

                        const image_t& image = task.image(sample.m_index);
                        const vector_t target = image.make_target(sample.m_region);
                        if (image.has_target(target))
                        {
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
                }

                std::cout << "vrad - end, count = " << lcount << std::endl;

                return lcount == 0 ? 0.0 : lvalue / lcount;
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
                // print model structure
                for (size_t l = 0; l < n_layers(); l ++)
                {
                        const conv_layer_t& layer = m_layers[l];

                        log_info() << "convolution model: layer ["
                                   << (l + 1) << "/" << n_layers() << "] "
                                   << layer.n_inputs() << "x" << layer.n_irows() << "x" << layer.n_icols()
                                   << " -> "
                                   << layer.n_outputs() << "x" << layer.n_orows() << "x" << layer.n_ocols() << ".";
                }

                // optimization problem: size
                auto opt_fn_size = [&] ()
                {
                        return n_parameters();
                };

                // optimization problem: function value
                auto opt_fn_fval = [&] (const vector_t& x)
                {
                        deserializer_t(x) >> m_layers;

                        return value(task, loss, samples);
                };

                // optimization problem: function value & gradient
                auto opt_fn_fval_grad = [&] (const vector_t& x, vector_t& gx)
                {
                        deserializer_t(x) >> m_layers;

                        const scalar_t fx = vgrad(task, loss, samples);

                        gx.resize(n_parameters());

                        serializer_t s(gx);
                        for (size_t l = 0; l < n_layers(); l ++)
                        {
                                const conv_layer_t& layer = m_layers[l];
                                s << layer.gdata();
                        }

                        return fx;
                };

                const optimize::problem_t problem(opt_fn_size, opt_fn_fval, opt_fn_fval_grad);

                // optimize
                static const size_t opt_iters = 256;
                static const scalar_t opt_eps = 1e-5;
                static const size_t opt_history = 8;

                std::cout << "n_parameters = " << n_parameters() << std::endl;

                vector_t x(n_parameters());

                std::cout << "layers size = "
                          << (m_layers[0].gdata().size() + m_layers[1].gdata().size()) << std::endl;

                serializer_t(x) << m_layers;

                std::cout << "x = [" << x.minCoeff() << ", " << x.maxCoeff() << "]" << std::endl;

                timer_t timer;
                const optimize::result_t res = optimize::lbfgs(
                        problem, x,
                        opt_iters, opt_eps, opt_history,
                        std::bind(update, _1, std::ref(timer)));

                deserializer_t(res.optimum().x) >> m_layers;

                // OK
                log_info() << "linear model: optimum [loss = " << res.optimum().f
                           << ", gradient = " << res.optimum().g.norm() << "]"
                           << ", iterations = [" << res.iterations() << "/" << opt_iters
                           << "], speed = [" << res.speed().avg() << " +/- " << res.speed().stdev() << "].";

                return true;
        }

        //-------------------------------------------------------------------------------------------------
}

