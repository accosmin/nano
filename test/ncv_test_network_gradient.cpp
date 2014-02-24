#include "ncv.h"
#include "models/forward_network.h"
#include <boost/program_options.hpp>
#include <set>

using namespace ncv;

static void test_grad(
        const string_t& header,
        const string_t& loss_id,
        model_t& model, vector_t& params, tensor_t& input, vector_t& target,
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

                input.random(random_t<scalar_t>(-0.2 / sqrt(n_parameters), +0.2 / sqrt(n_parameters)));

                vector_t gx_gd, gx_ax;
                problem_gd(x, gx_gd);
                problem_ax(x, gx_ax);

                const scalar_t dgx = (gx_gd - gx_ax).lpNorm<Eigen::Infinity>();

                log_info() << header << " [" << (t + 1) << "/" << n_tests
                           << "]: gradient accuracy = " << dgx << " (" << (dgx > 1e-6 ? "ERROR" : "OK") << ").";
        }
}

int main(int argc, char *argv[])
{
        ncv::init();

        const strings_t conv_layer_ids { "", "conv" };
        const strings_t pool_layer_ids { "" }; //, "smax-abs-pool", "smax-pool" };
        const strings_t full_layer_ids { "", "linear" };
        const strings_t actv_layer_ids { "", "unit", "tanh", "snorm" };        
        const strings_t loss_ids = loss_manager_t::instance().ids();

        const color_mode cmd_color = color_mode::luma;
        const size_t cmd_inputs = 1;
        const size_t cmd_irows = 12;
        const size_t cmd_icols = 12;
        const size_t cmd_outputs = 10;
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
                                                        random_t<size_t> rgen(2, 6);

                                                        string_t params;
                                                        params += "dims=" + text::to_string(rgen());
                                                        params += (rgen() % 2 == 0) ? ",rows=3,cols=3" : ",rows=5,cols=5";

                                                        desc += conv_layer_id + ":" + params + ";";
//                                                        desc += pool_layer_id + ";";
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
                network.resize(cmd_irows, cmd_icols, cmd_outputs, cmd_color);

                // build the inputs & outputs
                vector_t params(network.n_parameters());
                tensor_t sample(cmd_inputs, cmd_irows, cmd_icols);
                vector_t target(cmd_outputs);

                // test network
                for (const string_t& loss_id : loss_ids)
                {
                        test_grad("[loss = " + loss_id + "]", loss_id, network, params, sample, target, cmd_tests);
                }
        }

        // OK
        log_info() << done;
        return EXIT_SUCCESS;
}
