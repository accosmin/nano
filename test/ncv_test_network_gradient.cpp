#include "nanocv.h"
#include "models/forward_network.h"
#include <boost/program_options.hpp>
#include <set>

using namespace ncv;

static void test_grad(const string_t& header, const string_t& loss_id, model_t& model, size_t n_tests)
{
        const rloss_t loss = loss_manager_t::instance().get(loss_id);

        const size_t n_params = model.n_parameters();
        const size_t n_inputs = model.n_inputs() * model.n_rows() * model.n_cols();
        const size_t n_outputs = model.n_outputs();

        vector_t params(n_params);
        vector_t target(n_outputs);
        tensor_t inputs(model.n_inputs(), model.n_rows(), model.n_cols());

        // optimization problem (wrt parameters): size
        auto opt_fn_params_size = [&] ()
        {
                return n_params;
        };

        // optimization problem (wrt parameters): function value
        auto opt_fn_params_fval = [&] (const vector_t& x)
        {
                model.load_params(x);

                const vector_t outputs = model.value(inputs);

                return loss->value(target, outputs);
        };

        // optimization problem (wrt parameters): function value & gradient
        auto opt_fn_params_grad = [&] (const vector_t& x, vector_t& gx)
        {
                model.load_params(x);

                const vector_t outputs = model.value(inputs);

                vector_t grad_params, grad_inputs;
                model.gradient(loss->vgrad(target, outputs), grad_params, grad_inputs);
                gx = grad_params;

                return loss->value(target, outputs);
        };

        // optimization problem (wrt inputs): size
        auto opt_fn_inputs_size = [&] ()
        {
                return n_inputs;
        };

        // optimization problem (wrt inputs): function value
        auto opt_fn_inputs_fval = [&] (const vector_t& x)
        {
                model.load_params(params);

                tensor_t xinputs = inputs;
                xinputs.copy_from(x.data());
                const vector_t outputs = model.value(xinputs);

                return loss->value(target, outputs);
        };

        // optimization problem (wrt inputs): function value & gradient
        auto opt_fn_inputs_grad = [&] (const vector_t& x, vector_t& gx)
        {
                model.load_params(params);

                tensor_t xinputs = inputs;
                xinputs.copy_from(x.data());
                const vector_t outputs = model.value(xinputs);

                vector_t grad_params, grad_inputs;
                model.gradient(loss->vgrad(target, outputs), grad_params, grad_inputs);
                gx = grad_inputs;

                return loss->value(target, outputs);
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
