#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE "test_gradient"

#include <boost/test/unit_test.hpp>
#include "thread/pool.h"
#include "cortex/class.h"
#include "math/close.hpp"
#include "cortex/cortex.h"
#include "math/random.hpp"
#include "math/epsilon.hpp"
#include "tensor/random.hpp"
#include "text/to_string.hpp"
#include "cortex/optimizer.h"
#include "cortex/util/logger.h"
#include "cortex/accumulator.h"
#include <set>

namespace
{
        using namespace nano;

        const strings_t conv_layer_ids { "", "conv" };
        const strings_t pool_layer_ids { "", "pool-max", "pool-min", "pool-avg" };
        const strings_t full_layer_ids { "", "linear" };
        const strings_t actv_layer_ids { "", "act-unit", "act-tanh", "act-snorm", "act-splus" };

        const color_mode cmd_color = color_mode::rgba;
        const size_t cmd_irows = 8;
        const size_t cmd_icols = 8;
        const size_t cmd_outputs = 2;
        const size_t cmd_max_layers = 2;
        const size_t cmd_tests = 4;
        const size_t cmd_threads = 1;

        nano::random_t<size_t> urgen;                           // unbounded rng
        nano::random_t<scalar_t> prgen(-1.0, +1.0);             // rng for parameters
        nano::random_t<scalar_t> irgen(-0.1, +0.1);             // rng for inputs
        nano::random_t<tensor_size_t> trgen(0, cmd_outputs - 1);// rng for targets

        string_t make_model_description(
                size_t n_layers,
                const string_t& actv_layer_id,
                const string_t& pool_layer_id,
                const string_t& conv_layer_id,
                const string_t& full_layer_id)
        {
                string_t desc;

                // convolution part
                for (size_t l = 0; l < n_layers && !conv_layer_id.empty(); ++ l)
                {
                        nano::random_t<size_t> rgen(2, 3);

                        string_t params;
                        params += "dims=" + nano::to_string(rgen());
                        params += (rgen() % 2 == 0) ? ",rows=2,cols=2" : ",rows=3,cols=3";

                        desc += conv_layer_id + ":" + params + ";";
                        if (l == 0)
                        {
                                desc += pool_layer_id + ";";
                        }
                        desc += actv_layer_id + ";";
                }

                // fully-connected part
                for (size_t l = 0; l < n_layers && !full_layer_id.empty(); ++ l)
                {
                        nano::random_t<size_t> rgen(2, 3);

                        string_t params;
                        params += "dims=" + nano::to_string(rgen());

                        desc += full_layer_id + ":" + params + ";";
                        desc += actv_layer_id + ";";
                }

                desc += "linear:dims=" + nano::to_string(cmd_outputs) + ";";

                return desc;
        }

        auto make_grad_configs()
        {
                // evaluate the analytical gradient vs. the finite difference approximation for various:
                //      * convolution layers
                //      * convolution connection types
                //      * pooling layers
                //      * fully connected layers
                //      * activation layers
                std::set<string_t> descs;
                for (size_t n_layers = 0; n_layers <= cmd_max_layers; ++ n_layers)
                {
                        for (const auto& actv_layer_id : actv_layer_ids)
                        {
                                for (const auto& pool_layer_id : pool_layer_ids)
                                {
                                        for (const auto& conv_layer_id : conv_layer_ids)
                                        {
                                                for (const auto& full_layer_id : full_layer_ids)
                                                {
                                                        const auto desc = make_model_description(
                                                                n_layers,
                                                                actv_layer_id,
                                                                pool_layer_id,
                                                                conv_layer_id,
                                                                full_layer_id);

                                                        descs.insert(desc);
                                                }
                                        }
                                }
                        }
                }

                // create the <model description, loss id> configuration
                std::vector<std::pair<nano::string_t, nano::string_t> > result;
                for (const auto& desc : descs)
                {
                        const strings_t loss_ids = nano::get_losses().ids();
                        const string_t loss_id = loss_ids[urgen() % loss_ids.size()];

                        result.emplace_back(desc, loss_id);
                }

                // OK
                return result;
        }

        void update_fails(const string_t& header, const scalar_t delta,
                size_t& n_checks, size_t& n_fails1, size_t& n_fails2, size_t& n_fails3)
        {
                n_checks ++;
                if (!nano::close(delta, scalar_t(0), nano::epsilon1<scalar_t>()))
                {
                        n_fails1 ++;
                }
                if (!nano::close(delta, scalar_t(0), nano::epsilon2<scalar_t>()))
                {
                        n_fails2 ++;
                }
                if (!nano::close(delta, scalar_t(0), nano::epsilon3<scalar_t>()))
                {
                        n_fails3 ++;
                        BOOST_CHECK_LE(delta, nano::epsilon3<scalar_t>());
                        log_error() << header << ": error = " << delta << "/" << nano::epsilon3<scalar_t>() << "!";
                }
        }
}

namespace test
{
        using namespace nano;

        std::mutex mutex;

        size_t n_checks = 0;
        size_t n_fails1 = 0;
        size_t n_fails2 = 0;
        size_t n_fails3 = 0;

        void test_grad_params(const string_t& header, const string_t& loss_id, const model_t& model,
                accumulator_t& acc_params)
        {
                const rloss_t rloss = nano::get_losses().get(loss_id);
                const loss_t& loss = *rloss;

                const tensor_size_t psize = model.psize();
                const tensor_size_t osize = model.osize();

                vector_t params(psize);
                vector_t target(osize);
                tensor_t input(model.idims(), model.irows(), model.icols());

                // optimization problem (wrt parameters & inputs): size
                auto fn_params_size = [&] ()
                {
                        return psize;
                };

                // optimization problem (wrt parameters & inputs): function value
                auto fn_params_fval = [&] (const vector_t& x)
                {
                        acc_params.set_params(x);
                        acc_params.update(input, target, loss);

                        return acc_params.value();
                };

                // optimization problem (wrt parameters & inputs): function value & gradient
                auto fn_params_grad = [&] (const vector_t& x, vector_t& gx)
                {
                        acc_params.set_params(x);
                        acc_params.update(input, target, loss);

                        gx = acc_params.vgrad();
                        return acc_params.value();
                };

                // construct optimization problem: analytic gradient and finite difference approximation
                const opt_problem_t problem(fn_params_size, fn_params_fval, fn_params_grad);

                for (size_t t = 0; t < cmd_tests; ++ t)
                {
                        tensor::set_random(params, prgen);
                        tensor::set_random(input, irgen);
                        target = nano::class_target(trgen(), osize);

                        const scalar_t delta = problem.grad_accuracy(params);

                        // update statistics
                        const std::lock_guard<std::mutex> lock(test::mutex);

                        update_fails(header, delta, n_checks, n_fails1, n_fails2, n_fails3);
                }
        }

        void test_grad_params(const string_t& header, const string_t& loss_id, const model_t& model)
        {
                const strings_t criteria = nano::get_criteria().ids();
                const string_t criterion = criteria[urgen() % criteria.size()];

                accumulator_t acc_params(model, cmd_threads, criterion, criterion_t::type::vgrad, 0.1);
                test_grad_params(header + "[criterion = " + criterion + "]", loss_id, model, acc_params);
        }

        void test_grad_inputs(const string_t& header, const string_t& loss_id, const model_t& model)
        {
                const rmodel_t rmodel_inputs = model.clone();

                const rloss_t rloss = nano::get_losses().get(loss_id);
                const loss_t& loss = *rloss;

                const tensor_size_t psize = model.psize();
                const tensor_size_t isize = model.isize();
                const tensor_size_t osize = model.osize();

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

                for (size_t t = 0; t < cmd_tests; ++ t)
                {
                        tensor::set_random(params, prgen);
                        tensor::set_random(input, irgen);
                        target = nano::class_target(trgen(), osize);

                        vector_t analytic_inputs_grad, aproxdif_inputs_grad;

                        problem_analytic_inputs(input.vector(), analytic_inputs_grad);
                        problem_aproxdif_inputs(input.vector(), aproxdif_inputs_grad);

                        const scalar_t delta = (analytic_inputs_grad - aproxdif_inputs_grad).lpNorm<Eigen::Infinity>();

                        // update statistics
                        const std::lock_guard<std::mutex> lock(test::mutex);

                        update_fails(header, delta, n_checks, n_fails1, n_fails2, n_fails3);
                }
        }
}

BOOST_AUTO_TEST_CASE(test_gradient)
{
        using namespace nano;

        nano::init();

        const auto configs = ::make_grad_configs();

        // test each configuration
        nano::pool_t pool;
        for (const auto& config : configs)
        {
                pool.enqueue([=] ()
                {
                        const string_t desc = config.first;
                        const string_t loss_id = config.second;                        

                        {
                                const std::lock_guard<std::mutex> lock(test::mutex);
                                log_info() << desc;
                        }

                        // create model
                        const rmodel_t model = nano::get_models().get("forward-network", desc);
                        BOOST_CHECK_EQUAL(model.operator bool(), true);
                        {
                                const std::lock_guard<std::mutex> lock(test::mutex);

                                model->resize(cmd_irows, cmd_icols, cmd_outputs, cmd_color, false);
                                BOOST_CHECK_EQUAL(model->irows(), cmd_irows);
                                BOOST_CHECK_EQUAL(model->icols(), cmd_icols);
                                BOOST_CHECK_EQUAL(model->osize(), cmd_outputs);
                                BOOST_CHECK_EQUAL(static_cast<int>(model->color()), static_cast<int>(cmd_color));
                        }

                        // check with the given loss
                        test::test_grad_params("param [loss = " + loss_id + "]", loss_id, *model);
                        test::test_grad_inputs("input [loss = " + loss_id + "]", loss_id, *model);
                });
        };

        pool.wait();

        // check global results
        const scalar_t eps1 = nano::epsilon1<scalar_t>();
        const scalar_t eps2 = nano::epsilon2<scalar_t>();
        const scalar_t eps3 = nano::epsilon3<scalar_t>();

        log_info() << "failures: level1 = " << test::n_fails1 << "/" << test::n_checks << ", epsilon = " << eps1;
        log_info() << "failures: level2 = " << test::n_fails2 << "/" << test::n_checks << ", epsilon = " << eps2;
        log_info() << "failures: level3 = " << test::n_fails3 << "/" << test::n_checks << ", epsilon = " << eps3;

        BOOST_CHECK_LE(test::n_fails1 * 100, 100 * test::n_checks);
        BOOST_CHECK_LE(test::n_fails2 * 100, 5 * test::n_checks);
        BOOST_CHECK_LE(test::n_fails3 * 100, 0 * test::n_checks);
}
