#include "nanocv.h"
#include "accumulator.h"
#include "util/logger.h"
#include "util/random.hpp"
#include <boost/program_options.hpp>
#include <set>

using namespace ncv;

static void test_grad_params(
        const string_t& header, const string_t& loss_id, const model_t& model, accumulator_t& acc_params)
{
        random_t<size_t> rand(8, 16);

        const size_t n_tests = 64;
        const size_t n_samples = rand();

        const rloss_t rloss = loss_manager_t::instance().get(loss_id);
        const loss_t& loss = *rloss;

        const size_t psize = model.psize();
        const size_t isize = model.isize();
        const size_t osize = model.osize();

        vector_t params(psize);
        vectors_t targets(n_samples, vector_t(osize));
        tensors_t inputs(n_samples, tensor_t(model.idims(), model.irows(), model.icols()));

        // optimization problem (wrt parameters & inputs): size
        auto fn_params_size = [&] ()
        {
                return psize;
        };

        // optimization problem (wrt parameters & inputs): function value
        auto fn_params_fval = [&] (const vector_t& x)
        {
                acc_params.reset(x);
                acc_params.update(inputs, targets, loss);

                return acc_params.value();
        };

        // optimization problem (wrt parameters & inputs): function value & gradient
        auto fn_params_grad = [&] (const vector_t& x, vector_t& gx)
        {
                acc_params.reset(x);
                acc_params.update(inputs, targets, loss);

                gx = acc_params.vgrad();
                return acc_params.value();
        };

        // construct optimization problem: analytic gradient and finite difference approximation
        const opt_problem_t problem_analytic_params(fn_params_size, fn_params_fval, fn_params_grad);
        const opt_problem_t problem_aproxdif_params(fn_params_size, fn_params_fval);

        for (size_t t = 0; t < n_tests; t ++)
        {
                random_t<scalar_t> prgen(-1.0, +1.0);                
                random_t<scalar_t> irgen(-0.1, +0.1);
                random_t<size_t> trgen(0, osize - 1);

                prgen(params.data(), params.data() + psize);
                for (vector_t& target : targets)
                {
                        target = ncv::class_target(trgen(), osize);
                }
                for (tensor_t& input : inputs)
                {
                        irgen(input.data(), input.data() + isize);
                }

                vector_t analytic_params_grad, aproxdif_params_grad;

                problem_analytic_params(params, analytic_params_grad);
                problem_aproxdif_params(params, aproxdif_params_grad);

                const scalar_t dg_params = (analytic_params_grad - aproxdif_params_grad).lpNorm<Eigen::Infinity>();
                const bool ok = dg_params < 1e-5;//math::almost_equal(dg_params, scalar_t(0));

                log_info() << header << " [" << (t + 1) << "/" << n_tests
                           << "]: samples = " << n_samples
                           << ", dg_params = " << dg_params << " (" << (ok ? "OK" : "ERROR") << ").";
        }
}

static void test_grad_inputs(const string_t& header, const string_t& loss_id, const model_t& model)
{
        rmodel_t rmodel_inputs = model.clone();
        
        random_t<size_t> rand(8, 16);
        
        const size_t n_tests = 64;
        const size_t n_samples = rand();
        
        const rloss_t rloss = loss_manager_t::instance().get(loss_id);
        const loss_t& loss = *rloss;
        
        const size_t psize = model.psize();
        const size_t isize = model.isize();
        const size_t osize = model.osize();
        
        vector_t params(psize);
        vector_t target(osize);
        tensor_t input(model.idims(), model.irows(), model.icols());
        
        // optimization problem (wrt parameters & inputs): size
        auto fn_inputs_size = [&] ()
        {
                return isize;
        };
        
        // optimization problem (wrt parameters & inputs): function value
        auto fn_inputs_fval = [&] (const vector_t& x)
        {
                rmodel_inputs->load_params(params);
                
                const vector_t output = rmodel_inputs->output(x).vector();
                
                return loss.value(target, output);
        };
        
        // optimization problem (wrt parameters & inputs): function value & gradient
        auto fn_inputs_grad = [&] (const vector_t& x, vector_t& gx)
        {
                rmodel_inputs->load_params(params);
                
                const vector_t output = rmodel_inputs->output(x).vector();
                
                gx = rmodel_inputs->ginput(loss.vgrad(target, output)).vector();
                return loss.value(target, output);
        };
        
        // construct optimization problem: analytic gradient and finite difference approximation
        const opt_problem_t problem_analytic_inputs(fn_inputs_size, fn_inputs_fval, fn_inputs_grad);
        const opt_problem_t problem_aproxdif_inputs(fn_inputs_size, fn_inputs_fval);
        
        for (size_t t = 0; t < n_tests; t ++)
        {
                random_t<scalar_t> prgen(-1.0, +1.0);                
                random_t<scalar_t> irgen(-0.1, +0.1);
                random_t<size_t> trgen(0, osize - 1);
                
                prgen(params.data(), params.data() + psize);
                target = ncv::class_target(trgen(), osize);
                irgen(input.data(), input.data() + isize);
                
                vector_t analytic_inputs_grad, aproxdif_inputs_grad;
                
                problem_analytic_inputs(input.vector(), analytic_inputs_grad);
                problem_aproxdif_inputs(input.vector(), aproxdif_inputs_grad);
                
                const scalar_t dg_inputs = (analytic_inputs_grad - aproxdif_inputs_grad).lpNorm<Eigen::Infinity>();                
                const bool ok = dg_inputs < 1e-5;//math::almost_equal(dg_inputs, scalar_t(0));
                
                log_info() << header << " [" << (t + 1) << "/" << n_tests
                           << "]: samples = " << n_samples
                           << ", dg_inputs = " << dg_inputs << " (" << (ok ? "OK" : "ERROR") << ").";
        }
}

static void test_grad(const string_t& header, const string_t& loss_id, const model_t& model)
{
        // check all criteria
        const strings_t criteria = criterion_manager_t::instance().ids();
        for (const string_t& criterion : criteria)
        {
                random_t<size_t> rand(2, 16);
                const size_t n_threads = 1 + (rand() % 2);

                accumulator_t acc_params(model, n_threads, criterion, criterion_t::type::vgrad, 1.0);
                test_grad_params(header + "[criterion = " + criterion + "]", loss_id, model, acc_params);
        }
        
        // check gradients wrt the input
        test_grad_inputs(header, loss_id, model);
}

int main(int argc, char *argv[])
{
        ncv::init();

        const strings_t conv_layer_ids { "", "conv" };
        const strings_t conv_masks { "25", "50", "100" };
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
        //      * convolution connection types
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
                                        for (const string_t& conv_mask : conv_masks)
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
                                                                params += (rgen() % 2 == 0) ? ",rows=3,cols=3," : ",rows=4,cols=4,";
                                                                params += "mask=" + conv_mask;

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

                                                        descs.insert(desc);
                                                }
                                        }
                                }
                        }
                }
        }

        for (const string_t& desc : descs)
        {
                // create network
                const rmodel_t model = model_manager_t::instance().get("forward-network", desc);
                assert(model);
                model->resize(cmd_irows, cmd_icols, cmd_outputs, cmd_color, true);

                // test network
                for (const string_t& loss_id : loss_ids)
                {
                        test_grad("[loss = " + loss_id + "]", loss_id, *model);
                }
        }

        // OK
        log_info() << done;
        return EXIT_SUCCESS;
}
