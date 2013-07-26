#include "ncv.h"
#include "core/optimize.h"
#include "core/random.h"
#include "core/logger.h"
#include "model/conv_network.h"
#include "loss/loss_classnll.h"
#include <boost/program_options.hpp>

static void test(
        const ncv::string_t& header,
        const ncv::string_t& loss_id,
        ncv::conv_network_t& network, ncv::vector_t& params, ncv::tensor3d_t& input, ncv::vector_t& target,
        ncv::size_t n_tests)
{
        const ncv::rloss_t loss = ncv::loss_manager_t::instance().get(loss_id);
        const ncv::size_t n_parameters = params.size();

        // optimization problem: size
        auto opt_fn_size = [&] ()
        {
                return n_parameters;
        };

        // optimization problem: function value
        auto opt_fn_fval = [&] (const ncv::vector_t& x)
        {
                network.load_params(x);

                const ncv::vector_t output = network.value(input);

                return loss->value(target, output);
        };

        // optimization problem: function value & gradient
        auto opt_fn_fval_grad = [&] (const ncv::vector_t& x, ncv::vector_t& gx)
        {
                network.load_params(x);
                network.zero_grad();

                const ncv::vector_t output = network.value(input);
                network.cumulate_grad(loss->vgrad(target, output));
                gx = network.grad();

                return loss->value(target, output);
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

        const ncv::color_mode cmd_color = ncv::color_mode::luma;
        const ncv::size_t cmd_inputs = 1;
        const ncv::size_t cmd_irows = 16;
        const ncv::size_t cmd_icols = 16;
        const ncv::size_t cmd_outputs = 10;
        const ncv::size_t cmd_max_network = 3;

        const ncv::size_t cmd_tests = 8;

        // evaluate the analytical gradient vs. the finite difference approximation
        //      for each: loss, activation & number of network
        for (size_t n_network = 0; n_network <= cmd_max_network; n_network ++)
        {
                const ncv::strings_t _activation_ids =
                        (n_network == 0) ? ncv::strings_t(1, "unit") : activation_ids;

                for (const ncv::string_t& activation_id : _activation_ids)
                {
                        // build the convolution network
                        ncv::conv_layer_params_t network_params;
                        for (ncv::size_t l = 0; l < n_network; l ++)
                        {
                                ncv::random_t<ncv::size_t> rgen(2, 6);
                                network_params.push_back({rgen(), rgen(), rgen(), activation_id});
                        }

                        ncv::conv_network_t network(network_params);
                        network.resize(cmd_irows, cmd_icols, cmd_outputs, cmd_color);

                        // build the inputs & outputs
                        ncv::vector_t params(network.n_parameters());
                        ncv::tensor3d_t sample(cmd_inputs, cmd_irows, cmd_icols);
                        ncv::vector_t target(cmd_outputs);

                        // test network
                        for (const ncv::string_t& loss_id : loss_ids)
                        {
                                test("[network = " + ncv::text::to_string(n_network) +
                                     ", activation = " + activation_id +
                                     ", loss = " + loss_id + "]",
                                     loss_id, network, params, sample, target, cmd_tests);
                        }
                }
        }

        // OK
        ncv::log_info() << ncv::done;
        return EXIT_SUCCESS;
}
