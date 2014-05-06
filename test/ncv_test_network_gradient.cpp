#include "nanocv.h"
#include "models/forward_network.h"
#include <boost/program_options.hpp>
#include <set>

using namespace ncv;

static void test_grad(const string_t& header, const string_t& loss_id, const model_t& model, accumulator_t acc_params)
{
        random_t<size_t> rand(2, 16);
        const size_t n_tests = 64;
        const size_t n_samples = rand();

        const rloss_t rloss = loss_manager_t::instance().get(loss_id);
        const loss_t& loss = *rloss;

        const size_t n_params = model.psize();
        const size_t n_inputs = model.isize();
        const size_t n_outputs = model.osize();

        vector_t params(n_params);
        vectors_t targets(n_samples, vector_t(n_outputs));
        tensors_t inputs(n_samples, tensor_t(model.idims(), model.irows(), model.icols()));

        // optimization problem (wrt parameters): size
        auto opt_fn_params_size = [&] ()
        {
                return acc_params.dimensions();
        };

        // optimization problem (wrt parameters): function value
        auto opt_fn_params_fval = [&] (const vector_t& x)
        {
                acc_params.reset(x);
                acc_params.update(inputs, targets, loss);

                return acc_params.value();
        };

        // optimization problem (wrt parameters): function value & gradient
        auto opt_fn_params_grad = [&] (const vector_t& x, vector_t& gx)
        {
                acc_params.reset(x);
                acc_params.update(inputs, targets, loss);

                gx = acc_params.vgrad();
                return acc_params.value();
        };

        // construct optimization problem: analytic gradient and finite difference approximation
        const opt_problem_t problem_params_analytic(opt_fn_params_size, opt_fn_params_fval, opt_fn_params_grad);
        const opt_problem_t problem_params_aproxdif(opt_fn_params_size, opt_fn_params_fval);

        for (size_t t = 0; t < n_tests; t ++)
        {
                random_t<scalar_t> prgen(-1.0, +1.0);
                random_t<scalar_t> trgen(-1.0, +1.0);
                random_t<scalar_t> irgen(-0.1, +0.1);

                prgen(params.data(), params.data() + n_params);
                for (vector_t& target : targets)
                {
                        trgen(target.data(), target.data() + n_outputs);
                }
                for (tensor_t& input : inputs)
                {
                        irgen(input.data(), input.data() + n_inputs);
                }

                vector_t params_analytic_grad, params_aproxdif_grad;
                problem_params_analytic(params, params_analytic_grad);
                problem_params_aproxdif(params, params_aproxdif_grad);

                const scalar_t params_dgrad = (params_analytic_grad - params_aproxdif_grad).lpNorm<Eigen::Infinity>();

                const scalar_t eps = 1e-6;

                log_info() << header << " [" << (t + 1) << "/" << n_tests
                           << "]: samples = " << n_samples
                           << ", dgrad = " << params_dgrad
                           << " (" << (params_dgrad > eps ? "ERROR" : "OK") << ").";
        }
}

static void test_grad(const string_t& header, const string_t& loss_id, const model_t& model)
{
        std::vector<accumulator_t::regularizer> regularizers =
        {
                accumulator_t::regularizer::none,
                accumulator_t::regularizer::l2norm,
                accumulator_t::regularizer::variational
        };

        scalars_t lambdas =
        {
                1e-3, 1e-2, 1e-1, 1.0
        };

        // check regularizers
        for (auto regularizer : regularizers)
        {
                if (regularizer == accumulator_t::regularizer::none)
                {
                        test_grad(header, loss_id, model,
                        { model, accumulator_t::type::vgrad, regularizer, 0.0 });
                }

                else
                {
                        // check regularization weights
                        for (auto lambda : lambdas)
                        {
                                test_grad(header, loss_id, model,
                                { model, accumulator_t::type::vgrad, regularizer, lambda });
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
        const size_t cmd_irows = 10;
        const size_t cmd_icols = 10;
        const size_t cmd_outputs = 3;
        const size_t cmd_max_layers = 2;

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
                                                        random_t<size_t> rgen(2, 3);

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
                                                descs.insert(desc + "softmax;");
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
                        test_grad("[loss = " + loss_id + "]", loss_id, network);
                }
        }

        // OK
        log_info() << done;
        return EXIT_SUCCESS;
}
