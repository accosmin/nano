#include "nano.h"
#include "class.h"
#include "utest.h"
#include "math/random.h"
#include "math/numeric.h"
#include "math/epsilon.h"
#include "tensor/numeric.h"
#include "text/to_string.h"
#include "text/algorithm.h"
#include "layers/make_layers.h"

using namespace nano;

struct model_wrt_params_function_t final : public function_t
{
        model_wrt_params_function_t(const rmodel_t& model, const rloss_t& loss, const tensor3d_t& inputs, const vector_t& target) :
                function_t("model", model->psize(), model->psize(), model->psize(), convexity::no, 1e+6),
                m_model(model), m_loss(loss), m_inputs(inputs), m_target(target)
        {
        }

        scalar_t vgrad(const vector_t& x, vector_t* gx) const override
        {
                m_model->load(x);
                const vector_t output = m_model->output(m_inputs).vector();
                if (gx)
                {
                        *gx = m_model->gparam(m_loss->vgrad(m_target, output));
                }
                return m_loss->value(m_target, output);
        }

        const rmodel_t&         m_model;
        const rloss_t&          m_loss;
        const tensor3d_t&       m_inputs;
        const vector_t&         m_target;
};

struct model_wrt_inputs_function_t final : public function_t
{
        model_wrt_inputs_function_t(const rmodel_t& model, const rloss_t& loss, const vector_t& params, const vector_t& target) :
                function_t("model", nano::size(model->idims()), nano::size(model->idims()), nano::size(model->idims()), convexity::no, 1e+6),
                m_model(model), m_loss(loss), m_params(params), m_target(target)
        {
        }

        scalar_t vgrad(const vector_t& x, vector_t* gx) const override
        {
                m_model->load(m_params);
                const auto inputs = nano::map_tensor(x.data(), m_model->idims());
                const auto output = m_model->output(inputs).vector();
                if (gx)
                {
                        *gx = m_model->ginput(m_loss->vgrad(m_target, output)).vector();
                }
                return m_loss->value(m_target, output);
        }

        const rmodel_t&         m_model;
        const rloss_t&          m_loss;
        const vector_t&         m_params;
        const vector_t&         m_target;
};

const auto cmd_idims = dim3d_t{3, 8, 8};
const auto cmd_odims = dim3d_t{3, 1, 1};
const auto cmd_tests = size_t{27};
const auto cmd_layer_output = make_output_layer(cmd_odims);

static tensor_size_t apsize(const tensor_size_t isize, const tensor_size_t osize)
{
        return isize * osize + osize;
}

static tensor_size_t apsize(const dim3d_t& idims, const tensor_size_t osize)
{
        return apsize(nano::size(idims), osize);
}

static tensor_size_t apsize(const dim3d_t& idims, const dim3d_t& odims)
{
        return apsize(nano::size(idims), nano::size(odims));
}

static tensor_size_t apsize(const tensor_size_t isize, const dim3d_t& odims)
{
        return apsize(isize, nano::size(odims));
}

static tensor_size_t cpsize(const tensor_size_t idims,
        const tensor_size_t odims, const tensor_size_t krows, const tensor_size_t kcols, const tensor_size_t kconn)
{
        return idims * odims * krows * kcols / kconn + odims;
}

static tensor_size_t cpsize(const dim3d_t& idims,
        const tensor_size_t odims, const tensor_size_t krows, const tensor_size_t kcols, const tensor_size_t kconn)
{
        return cpsize(std::get<0>(idims), odims, krows, kcols, kconn);
}

static auto get_loss()
{
        static auto iloss = make_rng<size_t>();
        const auto loss_ids = get_losses().ids();
        const auto loss_id = loss_ids[iloss() % loss_ids.size()];

        return get_losses().get(loss_id);
}

static auto get_model(const string_t& description)
{
        auto model = get_models().get("forward-network", description + ";" + cmd_layer_output);
        model->resize(cmd_idims, cmd_odims);
        NANO_CHECK_EQUAL(model->idims(), cmd_idims);
        NANO_CHECK_EQUAL(model->odims(), cmd_odims);
        return model;
}

static void make_random_config(tensor3d_t& inputs, vector_t& target)
{
        random_t<scalar_t> irgen(-scalar_t(1.0), +scalar_t(1.0));
        random_t<tensor_size_t> trgen(0, target.size() - 1);

        nano::set_random(irgen, inputs);
        target = class_target(trgen(), target.size());
}

static void test_model(const string_t& model_description, const tensor_size_t expected_psize,
        const scalar_t epsilon = epsilon2<scalar_t>())
{
        const auto model = get_model(model_description);
        const auto loss = get_loss();

        NANO_CHECK_EQUAL(model->psize(), expected_psize);

        vector_t params(model->psize());
        vector_t target(nano::size(model->odims()));
        tensor3d_t inputs(model->idims());

        NANO_CHECK_EQUAL(model->odims(), cmd_odims);

        const model_wrt_params_function_t pfunction(model, loss, inputs, target);
        const model_wrt_inputs_function_t ifunction(model, loss, params, target);

        // construct optimization problem: analytic gradient vs finite difference approximation
        for (size_t t = 0; t < cmd_tests; ++ t)
        {
                make_random_config(inputs, target);
                model->random();
                NANO_CHECK(model->save(params));

                NANO_CHECK_LESS(pfunction.grad_accuracy(params), epsilon);
                NANO_CHECK_LESS(ifunction.grad_accuracy(inputs.vector()), epsilon);
        }
}

NANO_BEGIN_MODULE(test_layers)

NANO_CASE(affine0)
{
        test_model(
                "",
                apsize(cmd_idims, cmd_odims));
}

NANO_CASE(affine1)
{
        test_model(
                make_affine_layer(7),
                apsize(cmd_idims, 7) + apsize(7, cmd_odims));
}

NANO_CASE(activation)
{
        for (const auto& activation_id : get_layers().ids())
        {
                if (nano::starts_with(activation_id, "act-"))
                {
                        test_model(
                                activation_id,
                                apsize(cmd_idims, cmd_odims));
                }
        }
}

NANO_CASE(conv)
{
        test_model(
                make_conv_layer(3, 3, 3, 3, "act-unit"),
                cpsize(cmd_idims, 3, 3, 3, 3) + apsize(3 * 6 * 6, cmd_odims));

        test_model(
                make_conv_layer(4, 3, 3, 1, "act-snorm"),
                cpsize(cmd_idims, 4, 3, 3, 1) + apsize(4 * 6 * 6, cmd_odims));

        test_model(
                make_conv_layer(5, 3, 3, 1, "act-splus"),
                cpsize(cmd_idims, 5, 3, 3, 1) + apsize(5 * 6 * 6, cmd_odims));

        test_model
                (make_conv_layer(6, 3, 3, 3, "act-tanh"),
                cpsize(cmd_idims, 6, 3, 3, 3) + apsize(6 * 6 * 6, cmd_odims));
}

NANO_CASE(conv_stride)
{
        test_model(
                make_conv_layer(3, 5, 3, 3, "act-unit", 2, 1),
                cpsize(cmd_idims, 3, 5, 3, 3) + apsize(3 * 2 * 6, cmd_odims));

        test_model(
                make_conv_layer(3, 3, 5, 3, "act-snorm", 1, 2),
                cpsize(cmd_idims, 3, 3, 5, 3) + apsize(3 * 6 * 2, cmd_odims));

        test_model(
                make_conv_layer(3, 5, 5, 3, "act-splus", 2, 2),
                cpsize(cmd_idims, 3, 5, 5, 3) + apsize(3 * 2 * 2, cmd_odims));
}

NANO_CASE(multi_layer)
{
        test_model(
                make_affine_layer(7, "act-snorm") +
                make_affine_layer(5, "act-splus"),
                apsize(cmd_idims, 7) + apsize(7, 5) + apsize(5, cmd_odims));

        test_model(
                make_conv_layer(7, 3, 3, 1, "act-snorm") +
                make_conv_layer(4, 3, 3, 1, "act-splus"),
                cpsize(cmd_idims, 7, 3, 3, 1) + cpsize(7, 4, 3, 3, 1) + apsize(4 * 4 * 4, cmd_odims));

        test_model(
                make_conv_layer(7, 3, 3, 1, "act-snorm") +
                make_conv_layer(5, 3, 3, 1, "act-splus") +
                make_affine_layer(5, "act-splus"),
                cpsize(cmd_idims, 7, 3, 3, 1) + cpsize(7, 5, 3, 3, 1) + apsize(5 * 4 * 4, 5) + apsize(5, cmd_odims));

        test_model(
                make_conv_layer(8, 3, 3, 1, "act-snorm") +
                make_conv_layer(6, 3, 3, 2, "act-splus") +
                make_affine_layer(5, "act-splus"),
                cpsize(cmd_idims, 8, 3, 3, 1) + cpsize(8, 6, 3, 3, 2) + apsize(6 * 4 * 4, 5) + apsize(5, cmd_odims));

        test_model(
                make_conv_layer(8, 3, 3, 1, "act-snorm") +
                make_conv_layer(6, 3, 3, 2, "act-splus") +
                make_affine_layer(5, "act-splus"),
                cpsize(cmd_idims, 8, 3, 3, 1) + cpsize(8, 6, 3, 3, 2) + apsize(6 * 4 * 4, 5) + apsize(5, cmd_odims));

        test_model(
                make_conv_layer(9, 3, 3, 1, "act-snorm") +
                make_conv_layer(6, 3, 3, 3, "act-splus") +
                make_affine_layer(5, "act-splus"),
                cpsize(cmd_idims, 9, 3, 3, 1) + cpsize(9, 6, 3, 3, 3) + apsize(6 * 4 * 4, 5) + apsize(5, cmd_odims));
}

NANO_END_MODULE()
