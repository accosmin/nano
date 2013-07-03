#include "ncv.h"
#include "model/conv_layer.h"
#include "loss/loss_classnll.h"
#include <boost/program_options.hpp>

static const ncv::tensor3d_t& forward(ncv::conv_layers_t& layers, const ncv::tensor3d_t& _input)
{
        const ncv::tensor3d_t* input = &_input;
        for (ncv::conv_layers_t::const_iterator it = layers.begin(); it != layers.end(); ++ it)
        {
                input = &it->forward(*input);
        }

        return *input;
}

static void backward(ncv::conv_layers_t& layers, const ncv::tensor3d_t& _gradient)
{
        const ncv::tensor3d_t* gradient = &_gradient;
        for (ncv::conv_layers_t::reverse_iterator it = layers.rbegin(); it != layers.rend(); ++ it)
        {
                gradient = &it->backward(*gradient);
        }
}

static void test(
        const ncv::string_t& header,
        const ncv::string_t& loss_id,
        ncv::conv_layers_t& layers, ncv::vector_t& params, ncv::tensor3d_t& input, ncv::vector_t& target,
        ncv::size_t n_tests)
{
        const ncv::rloss_t loss = ncv::loss_manager_t::instance().get(loss_id, "");
        const ncv::size_t n_parameters = params.size();

        // optimization problem: size
        auto opt_fn_size = [&] ()
        {
                return n_parameters;
        };

        // optimization problem: function value
        auto opt_fn_fval = [&] (const ncv::vector_t& x)
        {
                ncv::deserializer_t(x) >> layers;

                // forward: network output
                const ncv::tensor3d_t& output = forward(layers, input);

                // loss value
                ncv::vector_t voutput(output.size());
                ncv::serializer_t(voutput) << output;

                return loss->value(target, voutput);
        };

        // optimization problem: function value & gradient
        auto opt_fn_fval_grad = [&] (const ncv::vector_t& x, ncv::vector_t& gx)
        {
                ncv::deserializer_t(x) >> layers;

                for (ncv::conv_layer_t& layer : layers)
                {
                        layer.zero_grad();
                }

                // forward: network output
                const ncv::tensor3d_t& output = forward(layers, input);

                // loss value & gradient
                ncv::vector_t voutput(output.size());
                ncv::serializer_t(voutput) << output;

                const ncv::scalar_t fx = loss->value(target, voutput);
                const ncv::vector_t vgradient = loss->vgrad(target, voutput);

                // backward: network gradient
                ncv::tensor3d_t gradient(output.size(), 1, 1);
                ncv::deserializer_t(vgradient) >> gradient;

                backward(layers, gradient);

                gx.resize(n_parameters);

                ncv::serializer_t s(gx);
                for (ncv::conv_layer_t& layer : layers)
                {
                        s << layer.gdata();
                }

                return fx;
        };

        // construct optimization problem: analytic gradient and finite difference approximation
        const ncv::optimize::problem_t problem_gd(opt_fn_size, opt_fn_fval, opt_fn_fval_grad);
        const ncv::optimize::problem_t problem_ax(opt_fn_size, opt_fn_fval);

        for (ncv::size_t t = 0; t < n_tests; t ++)
        {
                ncv::random_t<ncv::scalar_t> rgen(-1.0, +1.0);

                ncv::vector_t x(n_parameters);
                rgen(x.data(), x.data() + params.size());
                rgen(target.data(), target.data() + target.size());
                rgen(params.data(), params.data() + params.size());

                input.random(-0.1 / sqrt(n_parameters), 0.1 / sqrt(n_parameters));

                ncv::vector_t gx_gd, gx_ax;
                problem_gd.f(x, gx_gd);
                problem_ax.f(x, gx_ax);

                ncv::log_info() << header << " [" << (t + 1) << "/" << n_tests
                                << "]: gradient difference (analytic vs. finite difference) = "
                                << (gx_gd - gx_ax).lpNorm<Eigen::Infinity>() << ".";
        }
}

int main(int argc, char *argv[])
{
        ncv::init();

        const ncv::strings_t loss_ids = ncv::loss_manager_t::instance().ids();
        const ncv::strings_t activation_ids = ncv::activation_manager_t::instance().ids();

        const ncv::size_t cmd_inputs = 1;
        const ncv::size_t cmd_irows = 16;
        const ncv::size_t cmd_icols = 16;
        const ncv::size_t cmd_outputs = 10;
        const ncv::size_t cmd_max_layers = 3;

        const ncv::size_t cmd_tests = 8;

        // evaluate the analytical gradient vs. the finite difference approximation
        //      for each: loss, activation & number of layers
        for (size_t n_layers = 0; n_layers <= cmd_max_layers; n_layers ++)
        {
                const ncv::strings_t _activation_ids =
                        (n_layers == 0) ? ncv::strings_t(1, "unit") : activation_ids;

                for (const ncv::string_t& activation_id : _activation_ids)
                {
                        // build the convolution network
                        ncv::conv_layer_params_t network_params;
                        for (ncv::size_t l = 0; l < n_layers; l ++)
                        {
                                ncv::random_t<ncv::size_t> rgen(2, 6);
                                network_params.push_back({rgen(), rgen(), rgen(), activation_id});
                        }

                        ncv::conv_layers_t layers;
                        const ncv::size_t n_parameters = ncv::conv_layer_t::make_network(
                                cmd_inputs, cmd_irows, cmd_icols, network_params, cmd_outputs, layers);

                        ncv::log_info() << "number of parameters: " << n_parameters << ".";
                        ncv::conv_layer_t::print_network(layers);

                        // build the inputs & outputs
                        ncv::vector_t params(n_parameters);
                        ncv::tensor3d_t sample(cmd_inputs, cmd_irows, cmd_icols);
                        ncv::vector_t target(cmd_outputs);

                        // test network
                        for (const ncv::string_t& loss_id : loss_ids)
                        {
                                test("[layers = " + ncv::text::to_string(n_layers) +
                                     ", activation = " + activation_id +
                                     ", loss = " + loss_id + "]",
                                     loss_id, layers, params, sample, target, cmd_tests);
                        }
                }
        }

        // OK
        ncv::log_info() << ncv::done;
        return EXIT_SUCCESS;
}
