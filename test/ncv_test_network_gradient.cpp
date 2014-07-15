#include "nanocv.h"
#include "models/forward_network.h"
#include <boost/program_options.hpp>
#include <set>

using namespace ncv;

static void test_grad(const string_t& header, const string_t& loss_id, const model_t& model, scalar_t lambda)
{
        random_t<size_t> rand(2, 16);
        const size_t n_threads = 1 + (rand() % 2);

        accumulator_t acc_params(model, n_threads, accumulator_t::type::vgrad, lambda);
        
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
        auto opt_fn_size = [&] ()
        {
                return acc_params.dimensions();
        };

        // optimization problem (wrt parameters): function value
        auto opt_fn_fval = [&] (const vector_t& x)
        {
                acc_params.reset(x);
                acc_params.update(inputs, targets, loss);

                return acc_params.value();
        };

        // optimization problem (wrt parameters): function value & gradient
        auto opt_fn_grad = [&] (const vector_t& x, vector_t& gx)
        {
                acc_params.reset(x);
                acc_params.update(inputs, targets, loss);

                gx = acc_params.vgrad();
                return acc_params.value();
        };

        // construct optimization problem: analytic gradient and finite difference approximation
        const opt_problem_t problem_analytic(opt_fn_size, opt_fn_fval, opt_fn_grad);
        const opt_problem_t problem_aproxdif(opt_fn_size, opt_fn_fval);

        for (size_t t = 0; t < n_tests; t ++)
        {
                random_t<scalar_t> prgen(-1.0, +1.0);                
                random_t<scalar_t> irgen(-0.1, +0.1);
                random_t<size_t> trgen(0, n_outputs);

                prgen(params.data(), params.data() + n_params);
                for (vector_t& target : targets)
                {
                        target = ncv::class_target(trgen(), n_outputs);
                }
                for (tensor_t& input : inputs)
                {
                        irgen(input.data(), input.data() + n_inputs);
                }

                vector_t analytic_grad, aproxdif_grad;
                problem_analytic(params, analytic_grad);
                problem_aproxdif(params, aproxdif_grad);

                const scalar_t dgrad = (analytic_grad - aproxdif_grad).lpNorm<Eigen::Infinity>();
                const scalar_t eps = 1e-6;

                log_info() << header << " [" << (t + 1) << "/" << n_tests
                           << "]: samples = " << n_samples
                           << ", dgrad = " << dgrad << " (" << (dgrad > eps ? "ERROR" : "OK") << ").";
        }
}

static void test_grad(const string_t& header, const string_t& loss_id, const model_t& model)
{
        const scalars_t lambdas = { 0.0, 1e-3, 1e-2, 1e-1, 1.0 };
        for (scalar_t lambda : lambdas)
        {
                test_grad(header, loss_id, model, lambda);
        }
}

int main(int argc, char *argv[])
{
        ncv::init();

        const strings_t conv_layer_ids { "", "conv" };
        const strings_t pool_layer_ids { "", "pool-max", "pool-min", "pool-avg" };
        const strings_t full_layer_ids { "", "linear" };
        const strings_t actv_layer_ids { "", "act-unit", "act-tanh", "act-snorm", "act-splus" };
        const strings_t loss_ids = loss_manager_t::instance().ids();

        const color_mode cmd_color = color_mode::luma;
        const size_t cmd_irows = 10;
        const size_t cmd_icols = 10;
        const size_t cmd_outputs = 4;
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
                                                        random_t<size_t> rgen(1, 8);

                                                        string_t params;
                                                        params += "dims=" + text::to_string(rgen());

                                                        desc += full_layer_id + ":" + params + ";";
                                                        desc += actv_layer_id + ";";
                                                }

                                                desc += "linear:dims=" + text::to_string(cmd_outputs) + ";";
                                                desc += "softmax:type=global;";

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
                        test_grad("[loss = " + loss_id + "]", loss_id, network);
                }
        }

        // OK
        log_info() << done;
        return EXIT_SUCCESS;
}
