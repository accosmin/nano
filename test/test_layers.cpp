#include "loss.h"
#include "layer.h"
#include "model.h"
#include "utest.h"
#include "cortex.h"
#include "function.h"
#include "math/random.h"
#include "math/numeric.h"
#include "math/epsilon.h"
#include "tensor/numeric.h"
#include "text/algorithm.h"
#include "layers/make_layers.h"
#include "layers/conv_params.h"
#include "layers/affine_params.h"

using namespace nano;

struct model_wrt_params_function_t final : public function_t
{
        explicit model_wrt_params_function_t(const rloss_t& loss, const rmodel_t& model, const tensor_size_t count) :
                function_t("model", model->psize(), model->psize(), model->psize(), convexity::no, 1e+6),
                m_loss(loss),
                m_model(model),
                m_inputs(cat_dims(count, model->idims())),
                m_targets(cat_dims(count, model->odims()))
        {
                m_model->random();
                m_inputs.random(scalar_t(-0.1), scalar_t(+0.1));
                for (auto x = 0; x < count; ++ x)
                {
                        m_targets.vector(x) = class_target(x % model->osize(), model->osize());
                }
        }

        scalar_t vgrad(const vector_t& x, vector_t* gx) const override
        {
                NANO_CHECK_EQUAL(x.size(), m_model->psize());
                NANO_CHECK(std::isfinite(x.minCoeff()));
                NANO_CHECK(std::isfinite(x.maxCoeff()));

                m_model->params(x);
                const auto& outputs = m_model->output(m_inputs);
                NANO_CHECK(std::isfinite(outputs.vector().minCoeff()));
                NANO_CHECK(std::isfinite(outputs.vector().maxCoeff()));

                if (gx)
                {
                        const auto& gparam = m_model->gparam(m_loss->vgrad(m_targets, outputs));
                        NANO_CHECK_EQUAL(gx->size(), gparam.size());
                        NANO_CHECK(std::isfinite(gparam.vector().minCoeff()));
                        NANO_CHECK(std::isfinite(gparam.vector().maxCoeff()));

                        *gx = gparam.vector();
                }
                return m_loss->value(m_targets, outputs).vector().sum();
        }

        const rloss_t&          m_loss;
        const rmodel_t&         m_model;
        tensor4d_t              m_inputs;
        tensor4d_t              m_targets;
};

struct model_wrt_inputs_function_t final : public function_t
{
        explicit model_wrt_inputs_function_t(const rloss_t& loss, const rmodel_t& model, const tensor_size_t count) :
                function_t("model", count * model->isize(), count * model->isize(), count * model->isize(), convexity::no, 1e+6),
                m_loss(loss),
                m_model(model),
                m_inputs(cat_dims(count, model->idims())),
                m_targets(cat_dims(count, model->odims()))
        {
                m_model->random();
                m_inputs.random(scalar_t(-0.1), scalar_t(+0.1));
                for (auto x = 0; x < count; ++ x)
                {
                        m_targets.vector(x) = class_target(x % model->osize(), model->osize());
                }
        }

        scalar_t vgrad(const vector_t& x, vector_t* gx) const override
        {
                NANO_CHECK_EQUAL(x.size(), m_inputs.size());
                NANO_CHECK(std::isfinite(x.minCoeff()));
                NANO_CHECK(std::isfinite(x.maxCoeff()));

                m_inputs.vector() = x;
                const auto& outputs = m_model->output(m_inputs);
                NANO_CHECK(std::isfinite(outputs.vector().minCoeff()));
                NANO_CHECK(std::isfinite(outputs.vector().maxCoeff()));

                if (gx)
                {
                        const auto& ginput = m_model->ginput(m_loss->vgrad(m_targets, outputs));
                        NANO_CHECK_EQUAL(gx->size(), ginput.size());
                        NANO_CHECK(std::isfinite(ginput.vector().minCoeff()));
                        NANO_CHECK(std::isfinite(ginput.vector().maxCoeff()));

                        *gx = ginput.vector();
                }
                return m_loss->value(m_targets, outputs).vector().sum();
        }

        const rloss_t&          m_loss;
        const rmodel_t&         m_model;
        mutable tensor4d_t      m_inputs;
        tensor4d_t              m_targets;
};

const auto cmd_idims = tensor3d_dims_t{3, 6, 6};
const auto cmd_odims = tensor3d_dims_t{3, 1, 1};
const auto cmd_layer_output = make_output_layer(cmd_odims);

static tensor_size_t apsize(const tensor3d_dims_t& idims, const tensor3d_dims_t& odims)
{
        const auto params = affine_params_t{idims, odims};
        NANO_CHECK(params.valid());
        return params.psize();
}

static tensor_size_t cpsize(const tensor3d_dims_t& idims,
        const tensor_size_t omaps, const tensor_size_t krows, const tensor_size_t kcols, const tensor_size_t kconn)
{
        const auto params = conv_params_t{idims, omaps, kconn, krows, kcols};
        NANO_CHECK(params.valid());
        return params.psize();
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
        NANO_CHECK(model->configure(cmd_idims, cmd_odims));
        NANO_CHECK_EQUAL(model->idims(), cmd_idims);
        NANO_CHECK_EQUAL(model->odims(), cmd_odims);
        return model;
}

static void test_model(const string_t& model_description, const tensor_size_t expected_psize,
        const scalar_t epsilon = epsilon1<scalar_t>())
{
        const auto model = get_model(model_description);
        NANO_CHECK_EQUAL(model->psize(), expected_psize);

        for (auto count = 1; count <= 3; ++ count)
        {
                const auto loss = get_loss();
                const auto pfun = model_wrt_params_function_t{loss, model, count};
                const auto ifun = model_wrt_inputs_function_t{loss, model, count};

                const vector_t px = pfun.m_model->params();
                const vector_t ix = ifun.m_inputs.vector();

                NANO_CHECK_EQUAL(px.size(), pfun.size());
                NANO_CHECK_EQUAL(ix.size(), ifun.size());

                NANO_CHECK_LESS(pfun.grad_accuracy(px), epsilon);
                NANO_CHECK_LESS(ifun.grad_accuracy(ix), epsilon);
        }
}

NANO_BEGIN_MODULE(test_layers)

NANO_CASE(affine)
{
        test_model(
                "",
                apsize(cmd_idims, cmd_odims));

        test_model(
                make_affine_layer(7, 1, 1),
                apsize(cmd_idims, {7, 1, 1}) + apsize({7, 1, 1}, cmd_odims));
}

NANO_CASE(activation)
{
        for (const auto& layer_id : get_layers().ids())
        {
                if (is_activation_layer(layer_id))
                {
                        test_model(
                                make_layer(layer_id),
                                apsize(cmd_idims, cmd_odims));
                }
        }
}

NANO_CASE(conv3d)
{
        test_model(
                make_conv3d_layer(3, 3, 3, 3, "act-unit"),
                cpsize(cmd_idims, 3, 3, 3, 3) + apsize({3, 4, 4}, cmd_odims));

        test_model(
                make_conv3d_layer(4, 3, 3, 1, "act-snorm"),
                cpsize(cmd_idims, 4, 3, 3, 1) + apsize({4, 4, 4}, cmd_odims));

        test_model(
                make_conv3d_layer(5, 3, 3, 1, "act-splus"),
                cpsize(cmd_idims, 5, 3, 3, 1) + apsize({5, 4, 4}, cmd_odims));

        test_model
                (make_conv3d_layer(6, 3, 3, 3, "act-tanh"),
                cpsize(cmd_idims, 6, 3, 3, 3) + apsize({6, 4, 4}, cmd_odims));
}

NANO_CASE(conv3d_stride)
{
        test_model(
                make_conv3d_layer(3, 5, 3, 3, "act-unit", 2, 1),
                cpsize(cmd_idims, 3, 5, 3, 3) + apsize({3, 1, 4}, cmd_odims));

        test_model(
                make_conv3d_layer(3, 3, 5, 3, "act-snorm", 1, 2),
                cpsize(cmd_idims, 3, 3, 5, 3) + apsize({3, 4, 1}, cmd_odims));

        test_model(
                make_conv3d_layer(3, 5, 5, 3, "act-splus", 2, 2),
                cpsize(cmd_idims, 3, 5, 5, 3) + apsize({3, 1, 1}, cmd_odims));
}

NANO_CASE(norm_global_layer)
{
        test_model(
                make_norm_globally_layer(),
                apsize(cmd_idims, cmd_odims));
}

NANO_CASE(norm_plane_layer)
{
        test_model(
                make_norm_by_plane_layer(),
                apsize(cmd_idims, cmd_odims));
}

NANO_CASE(multi_layer)
{
        test_model(
                make_affine_layer(7, 1, 1, "act-snorm") +
                make_affine_layer(5, 1, 1, "act-splus"),
                apsize(cmd_idims, {7, 1, 1}) + apsize({7, 1, 1}, {5, 1, 1}) + apsize({5, 1, 1}, cmd_odims));

        test_model(
                make_conv3d_layer(7, 3, 3, 1, "act-snorm") +
                make_conv3d_layer(4, 1, 1, 1, "act-splus"),
                cpsize(cmd_idims, 7, 3, 3, 1) + cpsize({7, 4, 4}, 4, 1, 1, 1) + apsize({4, 4, 4}, cmd_odims));

        test_model(
                make_conv3d_layer(7, 3, 3, 1, "act-snorm") +
                make_conv3d_layer(4, 3, 3, 1, "act-splus"),
                cpsize(cmd_idims, 7, 3, 3, 1) + cpsize({7, 4, 4}, 4, 3, 3, 1) + apsize({4, 2, 2}, cmd_odims));

        test_model(
                make_conv3d_layer(7, 3, 3, 1, "act-snorm") +
                make_conv3d_layer(5, 3, 3, 1, "act-splus") +
                make_affine_layer(5, 1, 1, "act-splus"),
                cpsize(cmd_idims, 7, 3, 3, 1) + cpsize({7, 4, 4}, 5, 3, 3, 1) + apsize({5, 2, 2}, {5, 1, 1}) + apsize({5, 1, 1}, cmd_odims));

        test_model(
                make_conv3d_layer(8, 3, 3, 1, "act-snorm") +
                make_conv3d_layer(6, 3, 3, 2, "act-splus") +
                make_affine_layer(5, 1, 1, "act-splus"),
                cpsize(cmd_idims, 8, 3, 3, 1) + cpsize({8, 4, 4}, 6, 3, 3, 2) + apsize({6, 2, 2}, {5, 1, 1}) + apsize({5, 1, 1}, cmd_odims));

        test_model(
                make_conv3d_layer(9, 3, 3, 1, "act-snorm") +
                make_conv3d_layer(6, 3, 3, 3, "act-splus") +
                make_affine_layer(5, 1, 1, "act-splus"),
                cpsize(cmd_idims, 9, 3, 3, 1) + cpsize({9, 4, 4}, 6, 3, 3, 3) + apsize({6, 2, 2}, {5, 1, 1}) + apsize({5, 1, 1}, cmd_odims));
}

NANO_END_MODULE()
