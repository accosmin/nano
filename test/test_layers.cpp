#include "nano.h"
#include "class.h"
#include "utest.hpp"
#include "math/random.hpp"
#include "math/numeric.hpp"
#include "math/epsilon.hpp"
#include "tensor/numeric.hpp"
#include "text/to_string.hpp"
#include "layers/make_layers.h"

using namespace nano;

const tensor_size_t cmd_idims = 3;
const tensor_size_t cmd_irows = 8;
const tensor_size_t cmd_icols = 8;
const tensor_size_t cmd_isize = cmd_idims * cmd_irows * cmd_icols;
const tensor_size_t cmd_osize = 3;
const size_t cmd_tests = 7;

const string_t cmd_layer_output = make_output_layer(cmd_osize);

static tensor_size_t apsize(const tensor_size_t isize, const tensor_size_t osize)
{
        return isize * osize + osize;
}

static tensor_size_t cpsize(const tensor_size_t idims,
        const tensor_size_t odims, const tensor_size_t krows, const tensor_size_t kcols, const tensor_size_t kconn)
{
        return idims * odims * krows * kcols / kconn + odims;
}

static rloss_t get_loss()
{
        const strings_t loss_ids = get_losses().ids();

        const auto iloss = random_t<size_t>()();
        const auto loss_id = loss_ids[iloss % loss_ids.size()];

        return get_losses().get(loss_id);
}

static rmodel_t get_model(const string_t& description)
{
        const auto model = get_models().get("forward-network", description + ";" + cmd_layer_output);
        model->resize(cmd_idims, cmd_irows, cmd_icols, cmd_osize, false);
        NANO_CHECK_EQUAL(model->idims(), cmd_idims);
        NANO_CHECK_EQUAL(model->irows(), cmd_irows);
        NANO_CHECK_EQUAL(model->icols(), cmd_icols);
        NANO_CHECK_EQUAL(model->osize(), cmd_osize);
        return model;
}

static void make_random_config(tensor3d_t& inputs, vector_t& target)
{
        random_t<scalar_t> irgen(-scalar_t(1.0), +scalar_t(1.0));
        random_t<tensor_size_t> trgen(0, target.size() - 1);

        tensor::set_random(irgen, inputs);
        target = class_target(trgen(), target.size());
}

static void test_model(const string_t& model_description, const tensor_size_t expected_psize,
        const scalar_t epsilon = epsilon2<scalar_t>())
{
        const auto model = get_model(model_description);
        const auto loss = get_loss();

        NANO_CHECK_EQUAL(model->psize(), expected_psize);

        vector_t params(model->psize());
        vector_t target(model->osize());
        tensor3d_t inputs(model->idims(), model->irows(), model->icols());

        NANO_CHECK_EQUAL(model->osize(), cmd_osize);

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
                const auto output = model->output(tensor::map_tensor(x.data(), cmd_idims, cmd_irows, cmd_icols)).vector();

                return loss->value(target, output);
        };

        // optimization problem (wrt parameters & inputs): function value & gradient
        auto fn_inputs_grad = [&] (const vector_t& x, vector_t& gx)
        {
                model->load_params(params);
                const auto output = model->output(tensor::map_tensor(x.data(), cmd_idims, cmd_irows, cmd_icols)).vector();

                gx = model->ginput(loss->vgrad(target, output)).vector();
                return loss->value(target, output);
        };

        // construct optimization problem: analytic gradient vs finite difference approximation
        for (size_t t = 0; t < cmd_tests; ++ t)
        {
                make_random_config(inputs, target);
                model->random_params();
                NANO_CHECK(model->save_params(params));

                {
                        const problem_t problem(fn_params_size, fn_params_fval, fn_params_grad);
                        NANO_CHECK_LESS(problem.grad_accuracy(params), epsilon);
                }
                {
                        const problem_t problem(fn_inputs_size, fn_inputs_fval, fn_inputs_grad);
                        NANO_CHECK_LESS(problem.grad_accuracy(inputs.vector()), epsilon);
                }
        }
}

NANO_BEGIN_MODULE(test_layers)

NANO_CASE(activation)
{
        for (const auto& activation_id : { "act-unit", "act-tanh", "act-snorm", "act-splus" })
        {
                test_model(
                        activation_id,
                        apsize(cmd_isize, cmd_osize));
        }
}

NANO_CASE(affine)
{
        test_model(
                make_affine_layer(7),
                apsize(cmd_isize, 7) + apsize(7, cmd_osize));
}

NANO_CASE(conv)
{
        test_model(
                make_conv_layer(3, 3, 3, 1, "act-unit"),
                cpsize(cmd_idims, 3, 3, 3, 1) + apsize(3 * 6 * 6, cmd_osize));

        test_model(
                make_conv_layer(4, 3, 3, 1, "act-snorm"),
                cpsize(cmd_idims, 4, 3, 3, 1) + apsize(4 * 6 * 6, cmd_osize));

        test_model(
                make_conv_layer(5, 3, 3, 1, "act-splus"),
                cpsize(cmd_idims, 5, 3, 3, 1) + apsize(5 * 6 * 6, cmd_osize));

        test_model
                (make_conv_layer(6, 3, 3, 1, "act-tanh"),
                cpsize(cmd_idims, 6, 3, 3, 1) + apsize(6 * 6 * 6, cmd_osize));
}

NANO_CASE(conv_stride)
{
        test_model(
                make_conv_layer(3, 5, 3, 1, "act-unit", 2, 1),
                cpsize(cmd_idims, 3, 5, 3, 1) + apsize(3 * 2 * 6, cmd_osize));

        test_model(
                make_conv_layer(3, 3, 5, 1, "act-snorm", 1, 2),
                cpsize(cmd_idims, 3, 3, 5, 1) + apsize(3 * 6 * 2, cmd_osize));

        test_model(
                make_conv_layer(3, 5, 5, 1, "act-splus", 2, 2),
                cpsize(cmd_idims, 3, 5, 5, 1) + apsize(3 * 2 * 2, cmd_osize));
}

NANO_CASE(multi_layer)
{
        test_model(
                make_affine_layer(7, "act-snorm") +
                make_affine_layer(5, "act-splus"),
                apsize(cmd_isize, 7) + apsize(7, 5) + apsize(5, cmd_osize));

        test_model(
                make_conv_layer(7, 3, 3, 1, "act-snorm") +
                make_conv_layer(4, 3, 3, 1, "act-splus"),
                cpsize(cmd_idims, 7, 3, 3, 1) + cpsize(7, 4, 3, 3, 1) + apsize(4 * 4 * 4, cmd_osize));

        test_model(
                make_conv_layer(7, 3, 3, 1, "act-snorm") +
                make_conv_layer(5, 3, 3, 1, "act-splus") +
                make_affine_layer(5, "act-splus"),
                cpsize(cmd_idims, 7, 3, 3, 1) + cpsize(7, 5, 3, 3, 1) + apsize(5 * 4 * 4, 5) + apsize(5, cmd_osize));

        test_model(
                make_conv_layer(8, 3, 3, 1, "act-snorm") +
                make_conv_layer(6, 3, 3, 2, "act-splus") +
                make_affine_layer(5, "act-splus"),
                cpsize(cmd_idims, 8, 3, 3, 1) + cpsize(8, 6, 3, 3, 2) + apsize(6 * 4 * 4, 5) + apsize(5, cmd_osize));

        test_model(
                make_conv_layer(8, 3, 3, 1, "act-snorm") +
                make_conv_layer(6, 3, 3, 2, "act-splus") +
                make_affine_layer(5, "act-splus"),
                cpsize(cmd_idims, 8, 3, 3, 1) + cpsize(8, 6, 3, 3, 2) + apsize(6 * 4 * 4, 5) + apsize(5, cmd_osize));

        test_model(
                make_conv_layer(9, 3, 3, 1, "act-snorm") +
                make_conv_layer(6, 3, 3, 3, "act-splus") +
                make_affine_layer(5, "act-splus"),
                cpsize(cmd_idims, 9, 3, 3, 1) + cpsize(9, 6, 3, 3, 3) + apsize(6 * 4 * 4, 5) + apsize(5, cmd_osize));
}

NANO_END_MODULE()
