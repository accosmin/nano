#include "ncv.h"
#include "models/forward_network.h"
#include <boost/program_options.hpp>
#include <set>

using namespace ncv;

static void test_grad(
        const string_t& header,
        const string_t& loss_id,
        model_t& model, vector_t& params, tensor3d_t& input, vector_t& target,
        size_t n_tests)
{
        const rloss_t loss = loss_manager_t::instance().get(loss_id);
        const size_t n_parameters = params.size();

        // optimization problem: size
        auto opt_fn_size = [&] ()
        {
                return n_parameters;
        };

        // optimization problem: function value
        auto opt_fn_fval = [&] (const vector_t& x)
        {
                model.load_params(x);

                const vector_t output = model.value(input);

                return loss->value(target, output);
        };

        // optimization problem: function value & gradient
        auto opt_fn_fval_grad = [&] (const vector_t& x, vector_t& gx)
        {
                model.load_params(x);

                const vector_t output = model.value(input);
                gx = model.gradient(loss->vgrad(target, output));

                return loss->value(target, output);
        };

        // construct optimization problem: analytic gradient and finite difference approximation
        const opt_problem_t problem_gd(opt_fn_size, opt_fn_fval, opt_fn_fval_grad);
        const opt_problem_t problem_ax(opt_fn_size, opt_fn_fval);

        for (size_t t = 0; t < n_tests; t ++)
        {
                random_t<scalar_t> rgen(-1.0, +1.0);

                vector_t x(n_parameters);
                rgen(x.data(), x.data() + params.size());
                rgen(target.data(), target.data() + target.size());
                rgen(params.data(), params.data() + params.size());

                input.random(-0.1 / sqrt(n_parameters), +0.1 / sqrt(n_parameters));

                vector_t gx_gd, gx_ax;
                problem_gd(x, gx_gd);
                problem_ax(x, gx_ax);

                log_info() << header << " [" << (t + 1) << "/" << n_tests
                           << "]: gradient difference (analytic vs. finite difference) = "
                           << (gx_gd - gx_ax).lpNorm<Eigen::Infinity>() << ".";
        }
}

int main(int argc, char *argv[])
{
        ncv::init();

        const strings_t conv_layer_ids { "conv4x4" };
        const strings_t activation_layer_ids { "", "unit", "tanh", "snorm" };
        const strings_t loss_ids = loss_manager_t::instance().ids();

        const color_mode cmd_color = color_mode::luma;
        const size_t cmd_inputs = 1;
        const size_t cmd_irows = 16;
        const size_t cmd_icols = 16;
        const size_t cmd_outputs = 10;
        const size_t cmd_max_layers = 3;

        const size_t cmd_tests = 8;

        // evaluate the analytical gradient vs. the finite difference approximation
        //      for each: number of convolution layers, activation layer
        std::set<string_t> descs;
        for (size_t n_layers = 0; n_layers <= cmd_max_layers; n_layers ++)
        {
                for (const string_t& activation_layer_id : activation_layer_ids)
                {
                        for (const string_t& conv_layer_id : conv_layer_ids)
                        {
                                // build the network
                                string_t desc;
                                for (size_t l = 0; l < n_layers; l ++)
                                {
                                        random_t<size_t> rgen(2, 6);
                                        desc += conv_layer_id + ":convs=" + text::to_string(rgen()) + ";";
                                        desc += activation_layer_id + ";";
                                }

                                descs.insert(desc);
                        }
                }
        }

        for (const string_t& desc : descs)
        {                
                // create network
                forward_network_t network(desc);
                network.resize(cmd_irows, cmd_icols, cmd_outputs, cmd_color);

                // build the inputs & outputs
                vector_t params(network.n_parameters());
                tensor3d_t sample(cmd_inputs, cmd_irows, cmd_icols);
                vector_t target(cmd_outputs);

                // test network
                for (const string_t& loss_id : loss_ids)
                {
                        test_grad("[loss = " + loss_id + "]",
                                  loss_id, network, params, sample, target, cmd_tests);
                }
        }

        // OK
        log_info() << done;
        return EXIT_SUCCESS;
}
