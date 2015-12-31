#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE "test_layers"

#include <boost/test/unit_test.hpp>

#include "cortex/class.h"
#include "math/close.hpp"
#include "cortex/cortex.h"
#include "math/random.hpp"
#include "math/epsilon.hpp"
#include "tensor/random.hpp"
#include "cortex/optimizer.h"
#include "text/to_string.hpp"
#include "cortex/layers/make_layers.h"

using namespace cortex;

namespace 
{
        const color_mode cmd_color = color_mode::rgba;
        const size_t cmd_irows = 8;
        const size_t cmd_icols = 8;
        const size_t cmd_outputs = 3;
        const size_t cmd_tests = 7;

        const string_t cmd_layer_output = make_output_layer(cmd_outputs);

        rloss_t get_loss() 
        {
                const strings_t loss_ids = cortex::get_losses().ids();

                const auto iloss = math::random_t<size_t>()();
                const auto loss_id = loss_ids[iloss % loss_ids.size()];

                return cortex::get_losses().get(loss_id);
        }

        rmodel_t get_model(const string_t& description)
        {
                const auto model = cortex::get_models().get("forward-network", description + ";" + cmd_layer_output);
                model->resize(cmd_irows, cmd_icols, cmd_outputs, cmd_color, false);
                BOOST_CHECK_EQUAL(model->irows(), cmd_irows);
                BOOST_CHECK_EQUAL(model->icols(), cmd_icols);
                BOOST_CHECK_EQUAL(model->osize(), cmd_outputs);
                BOOST_CHECK_EQUAL(static_cast<int>(model->color()), static_cast<int>(cmd_color));
                return model;
        }

        void make_random_config(tensor_t& inputs, vector_t& params, vector_t& target)
        {
                math::random_t<scalar_t> irgen(-0.1, +0.1);
                math::random_t<scalar_t> prgen(-0.1, +0.1);
                math::random_t<tensor_size_t> trgen(0, target.size() - 1);

                tensor::set_random(inputs, irgen);
                tensor::set_random(params, prgen);
                target = cortex::class_target(trgen(), target.size());
        }

        void test_model(const string_t& model_description, const scalar_t epsilon = math::epsilon2<scalar_t>())
        {
                const auto model = get_model(model_description);
                const auto loss = get_loss();

                vector_t params(model->psize());
                vector_t target(model->osize());
                tensor_t inputs(model->idims(), model->irows(), model->icols());

                // optimization problem (wrt parameters & inputs): size
                auto fn_params_size = [&] ()
                {
                        return model->psize();
                };

                // optimization problem (wrt parameters & inputs): function value
                auto fn_params_fval = [&] (const vector_t& x)
                {
                        model->load_params(x);
                        const vector_t output = model->output(inputs).vector();

                        return loss->value(target, output);
                };

                // optimization problem (wrt parameters & inputs): function value & gradient
                auto fn_params_grad = [&] (const vector_t& x, vector_t& gx)
                {
                        model->load_params(x);
                        const vector_t output = model->output(inputs).vector();

                        gx = model->gparam(loss->vgrad(target, output));
                        return loss->value(target, output);
                };

                // optimization problem (wrt parameters & inputs): size
                auto fn_inputs_size = [&] ()
                {
                        return model->isize();
                };

                // optimization problem (wrt parameters & inputs): function value
                auto fn_inputs_fval = [&] (const vector_t& x)
                {
                        model->load_params(params);
                        const vector_t output = model->output(x).vector();

                        return loss->value(target, output);
                };

                // optimization problem (wrt parameters & inputs): function value & gradient
                auto fn_inputs_grad = [&] (const vector_t& x, vector_t& gx)
                {
                        model->load_params(params);
                        const vector_t output = model->output(x).vector();

                        gx = model->ginput(loss->vgrad(target, output)).vector();
                        return loss->value(target, output);
                };

                // construct optimization problem: analytic gradient vs finite difference approximation
                for (size_t t = 0; t < cmd_tests; ++ t)
                {
                        make_random_config(inputs, params, target);

                        {
                                const opt_problem_t problem(fn_params_size, fn_params_fval, fn_params_grad);
                                BOOST_CHECK_LE(problem.grad_accuracy(params), epsilon);
                        }
                        {
                                const opt_problem_t problem(fn_inputs_size, fn_inputs_fval, fn_inputs_grad);
                                BOOST_CHECK_LE(problem.grad_accuracy(inputs.vector()), epsilon);
                        }
                }
        }
}

BOOST_AUTO_TEST_CASE(test_activation)
{
        cortex::init();

        for (const auto& activation_id : { "act-unit", "act-tanh", "act-snorm", "act-splus" })
        {
                test_model(activation_id);
        }
}

BOOST_AUTO_TEST_CASE(test_affine)
{
        cortex::init();

        test_model(make_affine_layer(7));
}

BOOST_AUTO_TEST_CASE(test_conv)
{
        cortex::init();

        test_model(make_conv_pool_layer(3, 3, 3, "", ""));
        test_model(make_conv_pool_layer(3, 3, 3, "", "pool-max"));
        test_model(make_conv_pool_layer(3, 3, 3, "", "pool-min"));
        test_model(make_conv_pool_layer(3, 3, 3, "", "pool-avg"));
}

BOOST_AUTO_TEST_CASE(test_multi_layer_models)
{
        cortex::init();

        const strings_t descriptions =
        { 
                make_affine_layer(9, "act-snorm") +
                make_affine_layer(7, "act-splus"),

                make_conv_pool_layer(11, 3, 3, "act-snorm", "pool-max") +
                make_conv_layer(7, 3, 3, "act-splus"),

                make_conv_pool_layer(11, 3, 3, "act-snorm", "pool-max") +
                make_conv_layer(7, 3, 3, "act-splus") +
                make_affine_layer(9, "act-splus")
        };

        for (const auto& description : descriptions)
        {
                test_model(description);
        }
}
