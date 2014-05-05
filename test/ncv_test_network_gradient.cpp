#include "nanocv.h"
#include "models/forward_network.h"
#include <boost/program_options.hpp>
#include <set>

using namespace ncv;

static void test_grad(const string_t& header, const string_t& loss_id, const model_t& model, size_t n_tests,
                      accumulator_t acc_params, accumulator_t acc_inputs)
{
        const rloss_t rloss = loss_manager_t::instance().get(loss_id);
        const loss_t& loss = *rloss;

        const size_t n_params = model.n_parameters();
        const size_t n_inputs = model.n_inputs();
        const size_t n_outputs = model.n_outputs();

        vector_t params(n_params);
        vector_t target(n_outputs);
        tensor_t inputs(model.n_planes(), model.n_rows(), model.n_cols());

        // optimization problem (wrt parameters): size
        auto opt_fn_params_size = [&] ()
        {
                return acc_params.dimensions();
        };

        // optimization problem (wrt parameters): function value
        auto opt_fn_params_fval = [&] (const vector_t& x)
        {
                acc_params.reset(x);
                acc_params.update(inputs, target, loss);

                return acc_params.value();
        };

        // optimization problem (wrt parameters): function value & gradient
        auto opt_fn_params_grad = [&] (const vector_t& x, vector_t& gx)
        {
                acc_params.reset(x);
                acc_params.update(inputs, target, loss);

                gx = acc_params.vgrad();
                return acc_params.value();
        };

        // optimization problem (wrt inputs): size
        auto opt_fn_inputs_size = [&] ()
        {
                return acc_inputs.dimensions();
        };

        // optimization problem (wrt inputs): function value
        auto opt_fn_inputs_fval = [&] (const vector_t& x)
        {
                acc_inputs.reset(params);
                acc_inputs.update(x, target, loss);

                return acc_inputs.value();
        };

        // optimization problem (wrt inputs): function value & gradient
        auto opt_fn_inputs_grad = [&] (const vector_t& x, vector_t& gx)
        {
                acc_inputs.reset(params);
                acc_inputs.update(x, target, loss);

                gx = acc_inputs.vgrad();
                return acc_inputs.value();
        };

        // construct optimization problem: analytic gradient and finite difference approximation
        const opt_problem_t problem_params_analytic(opt_fn_params_size, opt_fn_params_fval, opt_fn_params_grad);
        const opt_problem_t problem_params_aproxdif(opt_fn_params_size, opt_fn_params_fval);

        const opt_problem_t problem_inputs_analytic(opt_fn_inputs_size, opt_fn_inputs_fval, opt_fn_inputs_grad);
        const opt_problem_t problem_inputs_aproxdif(opt_fn_inputs_size, opt_fn_inputs_fval);

        for (size_t t = 0; t < n_tests; t ++)
        {
                random_t<scalar_t> prgen(-1.0, +1.0);
                random_t<scalar_t> trgen(-1.0, +1.0);
                random_t<scalar_t> irgen(-0.1, +0.1);

                prgen(params.data(), params.data() + n_params);
                trgen(target.data(), target.data() + n_outputs);
                irgen(inputs.data(), inputs.data() + n_inputs);

                acc_params.reset(params);
                acc_inputs.reset(params);

                vector_t params_analytic_grad, params_aproxdif_grad;
                problem_params_analytic(params, params_analytic_grad);
                problem_params_aproxdif(params, params_aproxdif_grad);

                vector_t inputs_analytic_grad, inputs_aproxdif_grad;
                problem_inputs_analytic(inputs.vector(), inputs_analytic_grad);
                problem_inputs_aproxdif(inputs.vector(), inputs_aproxdif_grad);

                const scalar_t params_dgrad = (params_analytic_grad - params_aproxdif_grad).lpNorm<Eigen::Infinity>();
                const scalar_t inputs_dgrad = (inputs_analytic_grad - inputs_aproxdif_grad).lpNorm<Eigen::Infinity>();

                log_info() << header << " [" << (t + 1) << "/" << n_tests
                           << "]: gradient accuracy = " << params_dgrad << "/" << inputs_dgrad
                           << " (" << (std::max(params_dgrad, inputs_dgrad) > 1e-6 ? "ERROR" : "OK") << ").";
        }
}

static void test_grad(const string_t& header, const string_t& loss_id, const model_t& model, size_t n_tests)
{
        std::vector<accumulator_t::regularizer> regularizers =
        {
                accumulator_t::regularizer::none,
                accumulator_t::regularizer::l2norm
        };

        scalars_t lambdas =
        {
                0.0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0
        };

        // check reguaralizers
        for (auto regularizer : regularizers)
        {
                if (regularizer == accumulator_t::regularizer::none)
                {
                        test_grad(header, loss_id, model, n_tests,
                                { model, accumulator_t::type::vgrad, accumulator_t::source::params, regularizer, 0.0 },
                                { model, accumulator_t::type::vgrad, accumulator_t::source::inputs, accumulator_t::regularizer::none, 0.0 });
                }

                else
                {
                        // check regularization weights
                        for (auto lambda : lambdas)
                        {
                                test_grad(header, loss_id, model, n_tests,
                                        { model, accumulator_t::type::vgrad, accumulator_t::source::params, regularizer, lambda },
                                        { model, accumulator_t::type::vgrad, accumulator_t::source::inputs, accumulator_t::regularizer::none, lambda });
                        }
                }
        }
}

int main(int argc, char *argv[])
{
        ncv::init();

        const strings_t conv_layer_ids { "", "conv" };
        const strings_t pool_layer_ids { "", "smax-pool", "smax-abs-pool" };
        const strings_t full_layer_ids { "", "linear" };
        const strings_t actv_layer_ids { "", "unit", "tanh", "snorm" };        
        const strings_t loss_ids = { "classnll" };//loss_manager_t::instance().ids();

        const color_mode cmd_color = color_mode::luma;
        const size_t cmd_irows = 12;
        const size_t cmd_icols = 12;
        const size_t cmd_outputs = 4;
        const size_t cmd_max_layers = 2;

        const size_t cmd_tests = 64;

        // evaluate the analytical gradient vs. the finite difference approximation for various:
        //      * convolution layers
        //      * pooling layers
        //      * fully connected layers
        //      * activation layers
        std::set<string_t> descs;
        for (size_t n_layers = 0; n_layers <= cmd_max_layers; n_layers ++)
        {
                for (const string_t& actv_layer_id : actv_layer_ids)
                {
                        for (const string_t& pool_layer_id : pool_layer_ids)
                        {
                                for (const string_t& conv_layer_id : conv_layer_ids)
                                {
                                        for (const string_t& full_layer_id : full_layer_ids)
                                        {
                                                string_t desc;

                                                // convolution part
                                                for (size_t l = 0; l < n_layers && !conv_layer_id.empty(); l ++)
                                                {
                                                        random_t<size_t> rgen(2, 4);

                                                        string_t params;
                                                        params += "dims=" + text::to_string(rgen());
                                                        params += (rgen() % 2 == 0) ? ",rows=3,cols=3" : ",rows=4,cols=4";

                                                        desc += conv_layer_id + ":" + params + ";";
                                                        desc += pool_layer_id + ";";
                                                        desc += actv_layer_id + ";";
                                                }

                                                // fully-connected part
                                                for (size_t l = 0; l < n_layers && !full_layer_id.empty(); l ++)
                                                {
                                                        random_t<size_t> rgen(1, 10);

                                                        string_t params;
                                                        params += "dims=" + text::to_string(rgen());

                                                        desc += full_layer_id + ":" + params + ";";
                                                        desc += actv_layer_id + ";";
                                                }

                                                descs.insert(desc);
                                        }
                                }
                        }
                }
        }

        for (const string_t& desc : descs)
        {                
                // create network
                forward_network_t network(desc);
                network.resize(cmd_irows, cmd_icols, cmd_outputs, cmd_color, true);

                // test network
                for (const string_t& loss_id : loss_ids)
                {
                        test_grad("[loss = " + loss_id + "]", loss_id, network, cmd_tests);
                }
        }

        // OK
        log_info() << done;
        return EXIT_SUCCESS;
}
