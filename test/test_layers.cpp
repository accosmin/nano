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
#include "layers/builder.h"
#include "layers/conv3d_params.h"
#include "layers/affine_params.h"

using namespace nano;

struct model_wrt_params_function_t final : public function_t
{
        explicit model_wrt_params_function_t(const rloss_t& loss, model_t& model, const tensor_size_t count) :
                function_t("model", model.psize(), model.psize(), model.psize(), convexity::no, 1e+6),
                m_loss(loss),
                m_model(model),
                m_inputs(idims(model, count)),
                m_targets(odims(model, count))
        {
                m_model.random();
                m_inputs.random(0, 1);
                for (auto x = 0; x < count; ++ x)
                {
                        m_targets.vector(x) = class_target(x % osize(model), osize(model));
                }
        }

        scalar_t vgrad(const vector_t& x, vector_t* gx) const override
        {
                NANO_CHECK_EQUAL(x.size(), m_model.psize());
                NANO_CHECK(nano::isfinite(x));

                m_model.params(x);
                const auto& outputs = m_model.output(m_inputs);
                NANO_CHECK(nano::isfinite(outputs));

                if (gx)
                {
                        const auto& gparam = m_model.gparam(m_loss->vgrad(m_targets, outputs));
                        NANO_CHECK_EQUAL(gx->size(), gparam.size());
                        NANO_CHECK(nano::isfinite(gparam));

                        *gx = gparam;
                }
                return m_loss->value(m_targets, outputs).vector().sum();
        }

        const rloss_t&          m_loss;
        model_t&                m_model;
        tensor4d_t              m_inputs;
        tensor4d_t              m_targets;
};

struct model_wrt_inputs_function_t final : public function_t
{
        explicit model_wrt_inputs_function_t(const rloss_t& loss, model_t& model, const tensor_size_t count) :
                function_t("model", isize(model, count), isize(model, count), isize(model, count), convexity::no, 1e+6),
                m_loss(loss),
                m_model(model),
                m_inputs(idims(model, count)),
                m_targets(odims(model, count))
        {
                m_model.random();
                m_inputs.random(0, 1);
                for (auto x = 0; x < count; ++ x)
                {
                        m_targets.vector(x) = class_target(x % osize(model), osize(model));
                }
        }

        scalar_t vgrad(const vector_t& x, vector_t* gx) const override
        {
                NANO_CHECK_EQUAL(x.size(), m_inputs.size());
                NANO_CHECK(nano::isfinite(x));

                m_inputs.vector() = x;
                const auto& outputs = m_model.output(m_inputs);
                NANO_CHECK(nano::isfinite(outputs));

                if (gx)
                {
                        const auto& ginput = m_model.ginput(m_loss->vgrad(m_targets, outputs));
                        NANO_CHECK_EQUAL(gx->size(), ginput.size());
                        NANO_CHECK(nano::isfinite(ginput));

                        *gx = ginput.vector();
                }
                return m_loss->value(m_targets, outputs).vector().sum();
        }

        const rloss_t&          m_loss;
        model_t&                m_model;
        mutable tensor4d_t      m_inputs;
        tensor4d_t              m_targets;
};

const auto cmd_imaps = 3, cmd_irows = 6, cmd_icols = 6;
const auto cmd_omaps = 3, cmd_orows = 1, cmd_ocols = 1;
const auto cmd_idims = tensor3d_dim_t{cmd_imaps, cmd_irows, cmd_icols};
const auto cmd_odims = tensor3d_dim_t{cmd_omaps, cmd_orows, cmd_ocols};

static tensor_size_t apsize(const tensor3d_dim_t& idims, const tensor3d_dim_t& odims)
{
        const auto params = affine_params_t{idims, odims};
        NANO_CHECK(params.valid());
        return params.psize();
}

static tensor_size_t cpsize(const tensor3d_dim_t& idims,
        const tensor_size_t omaps, const tensor_size_t krows, const tensor_size_t kcols, const tensor_size_t kconn)
{
        const auto params = conv3d_params_t{idims, omaps, kconn, krows, kcols};
        NANO_CHECK(params.valid());
        return params.psize();
}

static void test_model(model_t& model, const tensor_size_t expected_psize,
        const scalar_t epsilon = 3 * epsilon1<scalar_t>())
{
        NANO_REQUIRE(model.done());
        NANO_REQUIRE(model.resize(cmd_idims, cmd_odims));
        NANO_CHECK_EQUAL(model.idims(), cmd_idims);
        NANO_CHECK_EQUAL(model.odims(), cmd_odims);
        NANO_CHECK_EQUAL(model.psize(), expected_psize);

        const auto count = 3;
        const auto loss = get_losses().get("s-logistic");
        const auto pfun = model_wrt_params_function_t{loss, model, count};
        const auto ifun = model_wrt_inputs_function_t{loss, model, count};

        const vector_t px = pfun.m_model.params();
        const vector_t ix = ifun.m_inputs.vector();

        NANO_CHECK_EQUAL(px.size(), pfun.size());
        NANO_CHECK_EQUAL(ix.size(), ifun.size());

        NANO_CHECK_LESS(pfun.grad_accuracy(px), epsilon);
        NANO_CHECK_LESS(ifun.grad_accuracy(ix), epsilon);
}

NANO_BEGIN_MODULE(test_layers)

NANO_CASE(affine)
{
        model_t model;
        NANO_CHECK(add_node(model, "1", affine_node_name(), config_affine_node, 7, 1, 1));
        NANO_CHECK(add_node(model, "2", "act-snorm", config_empty_node));
        NANO_CHECK(add_node(model, "3", affine_node_name(), config_affine_node, cmd_omaps, cmd_orows, cmd_ocols));
        NANO_CHECK(model.connect("1", "2", "3"));

        test_model(model,
                apsize(cmd_idims, {7, 1, 1}) + apsize({7, 1, 1}, cmd_odims));
}

NANO_CASE(activation)
{
        for (const auto& layer_id : get_layers().ids())
        {
                if (is_activation_node(layer_id))
                {
                        model_t model;
                        NANO_CHECK(add_node(model, "1", "act-snorm", config_empty_node));
                        NANO_CHECK(add_node(model, "2", affine_node_name(), config_affine_node, cmd_omaps, cmd_orows, cmd_ocols));
                        NANO_CHECK(model.connect("1", "2"));

                        test_model(model,
                                apsize(cmd_idims, cmd_odims));
                }
        }
}

NANO_CASE(conv3d1)
{
        model_t model;
        NANO_CHECK(add_node(model, "1", conv3d_node_name(), config_conv3d_node, 3, 3, 3, 3, 1, 1));
        NANO_CHECK(add_node(model, "2", "act-unit", config_empty_node));
        NANO_CHECK(add_node(model, "3", affine_node_name(), config_affine_node, cmd_omaps, cmd_orows, cmd_ocols));
        NANO_CHECK(model.connect("1", "2", "3"));

        test_model(model,
                cpsize(cmd_idims, 3, 3, 3, 3) + apsize({3, 4, 4}, cmd_odims));
}

NANO_CASE(conv3d2)
{
        model_t model;
        NANO_CHECK(add_node(model, "1", conv3d_node_name(), config_conv3d_node, 4, 3, 3, 1, 1, 1));
        NANO_CHECK(add_node(model, "2", "act-snorm", config_empty_node));
        NANO_CHECK(add_node(model, "3", affine_node_name(), config_affine_node, cmd_omaps, cmd_orows, cmd_ocols));
        NANO_CHECK(model.connect("1", "2", "3"));

        test_model(model,
                cpsize(cmd_idims, 4, 3, 3, 1) + apsize({4, 4, 4}, cmd_odims));
}

NANO_CASE(conv3d3)
{
        model_t model;
        NANO_CHECK(add_node(model, "1", conv3d_node_name(), config_conv3d_node, 5, 3, 3, 1, 1, 1));
        NANO_CHECK(add_node(model, "2", "act-splus", config_empty_node));
        NANO_CHECK(add_node(model, "3", affine_node_name(), config_affine_node, cmd_omaps, cmd_orows, cmd_ocols));
        NANO_CHECK(model.connect("1", "2", "3"));

        test_model(model,
                cpsize(cmd_idims, 5, 3, 3, 1) + apsize({5, 4, 4}, cmd_odims));
}

NANO_CASE(conv3d4)
{
        model_t model;
        NANO_CHECK(add_node(model, "1", conv3d_node_name(), config_conv3d_node, 6, 3, 3, 3, 1, 1));
        NANO_CHECK(add_node(model, "2", "act-tanh", config_empty_node));
        NANO_CHECK(add_node(model, "3", affine_node_name(), config_affine_node, cmd_omaps, cmd_orows, cmd_ocols));
        NANO_CHECK(model.connect("1", "2", "3"));

        test_model(model,
                cpsize(cmd_idims, 6, 3, 3, 3) + apsize({6, 4, 4}, cmd_odims));
}

NANO_CASE(conv3d_stride1)
{
        model_t model;
        NANO_CHECK(add_node(model, "1", conv3d_node_name(), config_conv3d_node, 3, 5, 3, 3, 2, 1));
        NANO_CHECK(add_node(model, "2", "act-unit", config_empty_node));
        NANO_CHECK(add_node(model, "3", affine_node_name(), config_affine_node, cmd_omaps, cmd_orows, cmd_ocols));
        NANO_CHECK(model.connect("1", "2", "3"));

        test_model(model,
                cpsize(cmd_idims, 3, 5, 3, 3) + apsize({3, 1, 4}, cmd_odims));
}

NANO_CASE(conv3d_stride2)
{
        model_t model;
        NANO_CHECK(add_node(model, "1", conv3d_node_name(), config_conv3d_node, 3, 3, 5, 3, 1, 2));
        NANO_CHECK(add_node(model, "2", "act-snorm", config_empty_node));
        NANO_CHECK(add_node(model, "3", affine_node_name(), config_affine_node, cmd_omaps, cmd_orows, cmd_ocols));
        NANO_CHECK(model.connect("1", "2", "3"));

        test_model(model,
                cpsize(cmd_idims, 3, 3, 5, 3) + apsize({3, 4, 1}, cmd_odims));
}

NANO_CASE(conv3d_stride3)
{
        model_t model;
        NANO_CHECK(add_node(model, "1", conv3d_node_name(), config_conv3d_node, 3, 5, 5, 3, 2, 2));
        NANO_CHECK(add_node(model, "2", "act-splus", config_empty_node));
        NANO_CHECK(add_node(model, "3", affine_node_name(), config_affine_node, cmd_omaps, cmd_orows, cmd_ocols));
        NANO_CHECK(model.connect("1", "2", "3"));

        test_model(model,
                cpsize(cmd_idims, 3, 5, 5, 3) + apsize({3, 1, 1}, cmd_odims));
}

NANO_CASE(norm_global_layer)
{
        model_t model;
        NANO_CHECK(add_node(model, "1", norm3d_node_name(), config_norm3d_node, norm_type::global));
        NANO_CHECK(add_node(model, "2", "act-snorm", config_empty_node));
        NANO_CHECK(add_node(model, "3", affine_node_name(), config_affine_node, cmd_omaps, cmd_orows, cmd_ocols));
        NANO_CHECK(model.connect("1", "2", "3"));

        test_model(model,
                apsize(cmd_idims, cmd_odims));
}

NANO_CASE(norm_plane_layer)
{
        model_t model;
        NANO_CHECK(add_node(model, "1", norm3d_node_name(), config_norm3d_node, norm_type::plane));
        NANO_CHECK(add_node(model, "2", "act-snorm", config_empty_node));
        NANO_CHECK(add_node(model, "3", affine_node_name(), config_affine_node, cmd_omaps, cmd_orows, cmd_ocols));
        NANO_CHECK(model.connect("1", "2", "3"));

        test_model(model,
                apsize(cmd_idims, cmd_odims));
}

NANO_CASE(multi_layer0)
{
        model_t model;
        NANO_CHECK(add_node(model, "1", affine_node_name(), config_affine_node, 7, 1, 1));
        NANO_CHECK(add_node(model, "2", "act-snorm", config_empty_node));
        NANO_CHECK(add_node(model, "3", affine_node_name(), config_affine_node, 5, 1, 1));
        NANO_CHECK(add_node(model, "4", "act-splus", config_empty_node));
        NANO_CHECK(add_node(model, "5", affine_node_name(), config_affine_node, cmd_omaps, cmd_orows, cmd_ocols));
        NANO_CHECK(model.connect("1", "2", "3", "4", "5"));

        test_model(model,
                apsize(cmd_idims, {7, 1, 1}) + apsize({7, 1, 1}, {5, 1, 1}) + apsize({5, 1, 1}, cmd_odims));
}

NANO_CASE(multi_layer1)
{
        model_t model;
        NANO_CHECK(add_node(model, "1", conv3d_node_name(), config_conv3d_node, 7, 3, 3, 1, 1, 1));
        NANO_CHECK(add_node(model, "2", "act-snorm", config_empty_node));
        NANO_CHECK(add_node(model, "3", conv3d_node_name(), config_conv3d_node, 4, 1, 1, 1, 1, 1));
        NANO_CHECK(add_node(model, "4", "act-splus", config_empty_node));
        NANO_CHECK(add_node(model, "5", affine_node_name(), config_affine_node, cmd_omaps, cmd_orows, cmd_ocols));
        NANO_CHECK(model.connect("1", "2", "3", "4", "5"));

        test_model(model,
                cpsize(cmd_idims, 7, 3, 3, 1) + cpsize({7, 4, 4}, 4, 1, 1, 1) + apsize({4, 4, 4}, cmd_odims));
}

NANO_CASE(multi_layer2)
{
        model_t model;
        NANO_CHECK(add_node(model, "1", conv3d_node_name(), config_conv3d_node, 7, 3, 3, 1, 1, 1));
        NANO_CHECK(add_node(model, "2", "act-snorm", config_empty_node));
        NANO_CHECK(add_node(model, "3", conv3d_node_name(), config_conv3d_node, 4, 3, 3, 1, 1, 1));
        NANO_CHECK(add_node(model, "4", "act-splus", config_empty_node));
        NANO_CHECK(add_node(model, "5", affine_node_name(), config_affine_node, cmd_omaps, cmd_orows, cmd_ocols));
        NANO_CHECK(model.connect("1", "2", "3", "4", "5"));

        test_model(model,
                cpsize(cmd_idims, 7, 3, 3, 1) + cpsize({7, 4, 4}, 4, 3, 3, 1) + apsize({4, 2, 2}, cmd_odims));
}

NANO_CASE(multi_layer3)
{
        model_t model;
        NANO_CHECK(add_node(model, "1", conv3d_node_name(), config_conv3d_node, 7, 3, 3, 1, 1, 1));
        NANO_CHECK(add_node(model, "2", "act-snorm", config_empty_node));
        NANO_CHECK(add_node(model, "3", conv3d_node_name(), config_conv3d_node, 5, 3, 3, 1, 1, 1));
        NANO_CHECK(add_node(model, "4", "act-splus", config_empty_node));
        NANO_CHECK(add_node(model, "5", affine_node_name(), config_affine_node, 5, 1, 1));
        NANO_CHECK(add_node(model, "6", "act-splus", config_empty_node));
        NANO_CHECK(add_node(model, "7", affine_node_name(), config_affine_node, cmd_omaps, cmd_orows, cmd_ocols));
        NANO_CHECK(model.connect("1", "2", "3", "4", "5", "6", "7"));

        test_model(model,
                cpsize(cmd_idims, 7, 3, 3, 1) + cpsize({7, 4, 4}, 5, 3, 3, 1) + apsize({5, 2, 2}, {5, 1, 1}) + apsize({5, 1, 1}, cmd_odims));
}

NANO_CASE(multi_layer4)
{
        model_t model;
        NANO_CHECK(add_node(model, "1", conv3d_node_name(), config_conv3d_node, 8, 3, 3, 1, 1, 1));
        NANO_CHECK(add_node(model, "2", "act-snorm", config_empty_node));
        NANO_CHECK(add_node(model, "3", conv3d_node_name(), config_conv3d_node, 6, 3, 3, 2, 1, 1));
        NANO_CHECK(add_node(model, "4", "act-splus", config_empty_node));
        NANO_CHECK(add_node(model, "5", affine_node_name(), config_affine_node, 5, 1, 1));
        NANO_CHECK(add_node(model, "6", "act-splus", config_empty_node));
        NANO_CHECK(add_node(model, "7", affine_node_name(), config_affine_node, cmd_omaps, cmd_orows, cmd_ocols));
        NANO_CHECK(model.connect("1", "2", "3", "4", "5", "6", "7"));

        test_model(model,
                cpsize(cmd_idims, 8, 3, 3, 1) + cpsize({8, 4, 4}, 6, 3, 3, 2) + apsize({6, 2, 2}, {5, 1, 1}) + apsize({5, 1, 1}, cmd_odims));
}

NANO_CASE(multi_layer5)
{
        model_t model;
        NANO_CHECK(add_node(model, "1", conv3d_node_name(), config_conv3d_node, 9, 3, 3, 1, 1, 1));
        NANO_CHECK(add_node(model, "2", "act-snorm", config_empty_node));
        NANO_CHECK(add_node(model, "3", conv3d_node_name(), config_conv3d_node, 6, 3, 3, 3, 1, 1));
        NANO_CHECK(add_node(model, "4", "act-splus", config_empty_node));
        NANO_CHECK(add_node(model, "5", affine_node_name(), config_affine_node, 5, 1, 1));
        NANO_CHECK(add_node(model, "6", "act-splus", config_empty_node));
        NANO_CHECK(add_node(model, "7", affine_node_name(), config_affine_node, cmd_omaps, cmd_orows, cmd_ocols));
        NANO_CHECK(model.connect("1", "2", "3", "4", "5", "6", "7"));

        test_model(model,
                cpsize(cmd_idims, 9, 3, 3, 1) + cpsize({9, 4, 4}, 6, 3, 3, 3) + apsize({6, 2, 2}, {5, 1, 1}) + apsize({5, 1, 1}, cmd_odims));
}

NANO_CASE(multi_mix_plus4d)
{
        model_t model;
        NANO_CHECK(add_node(model, "00", norm3d_node_name(), config_norm3d_node, norm_type::plane));
        NANO_CHECK(add_node(model, "11", conv3d_node_name(), config_conv3d_node, 4, 5, 5, 1, 1, 1));
        NANO_CHECK(add_node(model, "12", "act-snorm", config_empty_node));
        NANO_CHECK(add_node(model, "21", conv3d_node_name(), config_conv3d_node, 3, 3, 3, 1, 1, 1));
        NANO_CHECK(add_node(model, "22", "act-snorm", config_empty_node));
        NANO_CHECK(add_node(model, "23", conv3d_node_name(), config_conv3d_node, 4, 3, 3, 1, 1, 1));
        NANO_CHECK(add_node(model, "24", "act-snorm", config_empty_node));
        NANO_CHECK(add_node(model, "xx", mix_plus4d_node_name(), config_empty_node));
        NANO_CHECK(add_node(model, "x1", "act-splus", config_empty_node));
        NANO_CHECK(add_node(model, "x2", affine_node_name(), config_affine_node, 5, 1, 1));
        NANO_CHECK(add_node(model, "x3", "act-splus", config_empty_node));
        NANO_CHECK(add_node(model, "x4", affine_node_name(), config_affine_node, cmd_omaps, cmd_orows, cmd_ocols));
        NANO_CHECK(model.connect("00", "11", "12", "xx"));
        NANO_CHECK(model.connect("00", "21", "22", "23", "24", "xx"));
        NANO_CHECK(model.connect("xx", "x1", "x2", "x3", "x4"));

        test_model(model,
                cpsize(cmd_idims, 4, 5, 5, 1) +
                cpsize(cmd_idims, 3, 3, 3, 1) + cpsize({3, 4, 4}, 4, 3, 3, 1) +
                apsize({4, 2, 2}, {5, 1, 1}) + apsize({5, 1, 1}, cmd_odims));
}

NANO_CASE(multi_mix_tcat4d)
{
        model_t model;
        NANO_CHECK(add_node(model, "00", norm3d_node_name(), config_norm3d_node, norm_type::plane));
        NANO_CHECK(add_node(model, "11", conv3d_node_name(), config_conv3d_node, 4, 5, 5, 1, 1, 1));
        NANO_CHECK(add_node(model, "12", "act-snorm", config_empty_node));
        NANO_CHECK(add_node(model, "21", conv3d_node_name(), config_conv3d_node, 3, 3, 3, 1, 1, 1));
        NANO_CHECK(add_node(model, "22", "act-snorm", config_empty_node));
        NANO_CHECK(add_node(model, "23", conv3d_node_name(), config_conv3d_node, 4, 3, 3, 1, 1, 1));
        NANO_CHECK(add_node(model, "24", "act-snorm", config_empty_node));
        NANO_CHECK(add_node(model, "xx", mix_tcat4d_node_name(), config_empty_node));
        NANO_CHECK(add_node(model, "x1", "act-splus", config_empty_node));
        NANO_CHECK(add_node(model, "x2", affine_node_name(), config_affine_node, 5, 1, 1));
        NANO_CHECK(add_node(model, "x3", "act-splus", config_empty_node));
        NANO_CHECK(add_node(model, "x4", affine_node_name(), config_affine_node, cmd_omaps, cmd_orows, cmd_ocols));
        NANO_CHECK(model.connect("00", "11", "12", "xx"));
        NANO_CHECK(model.connect("00", "21", "22", "23", "24", "xx"));
        NANO_CHECK(model.connect("xx", "x1", "x2", "x3", "x4"));

        test_model(model,
                cpsize(cmd_idims, 4, 5, 5, 1) +
                cpsize(cmd_idims, 3, 3, 3, 1) + cpsize({3, 4, 4}, 4, 3, 3, 1) +
                apsize({8, 2, 2}, {5, 1, 1}) + apsize({5, 1, 1}, cmd_odims));
}

NANO_END_MODULE()
