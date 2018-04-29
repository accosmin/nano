#include "loss.h"
#include "utest.h"
#include "cortex.h"
#include "builder.h"
#include "function.h"
#include "math/random.h"
#include "math/numeric.h"
#include "math/epsilon.h"
#include "tensor/numeric.h"
#include "text/algorithm.h"
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
                NANO_CHECK(x.array().isFinite().all());

                m_model.params(x);
                const auto& outputs = m_model.output(m_inputs);
                NANO_CHECK(outputs.array().isFinite().all());

                if (gx)
                {
                        const auto& gparam = m_model.gparam(m_loss->vgrad(m_targets, outputs));
                        NANO_CHECK_EQUAL(gx->size(), gparam.size());
                        NANO_CHECK(gparam.array().isFinite().all());

                        *gx = gparam;
                }
                return m_loss->value(m_targets, outputs).vector().sum();
        }

        const rloss_t&          m_loss;
        model_t&                m_model;
        tensor4d_t              m_inputs;
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
        const scalar_t epsilon = epsilon2<scalar_t>())
{
        NANO_REQUIRE(model.done());
        NANO_REQUIRE(model.resize(cmd_idims, cmd_odims));
        NANO_CHECK_EQUAL(model.idims(), cmd_idims);
        NANO_CHECK_EQUAL(model.odims(), cmd_odims);
        NANO_CHECK_EQUAL(model.psize(), expected_psize);

        const auto count = 3;
        const auto loss = get_losses().get("s-logistic");
        const auto pfun = model_wrt_params_function_t{loss, model, count};

        const vector_t px = pfun.m_model.params();
        NANO_CHECK_EQUAL(px.size(), pfun.size());
        NANO_CHECK_LESS(pfun.grad_accuracy(px), epsilon);
}

NANO_BEGIN_MODULE(test_layers)

NANO_CASE(affine)
{
        model_t model;
        NANO_CHECK(model.add(config_affine_node("1", 7, 1, 1)));
        NANO_CHECK(model.add(config_activation_node("2", "act-snorm")));
        NANO_CHECK(model.add(config_affine_node("3", cmd_omaps, cmd_orows, cmd_ocols)));
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
                        NANO_CHECK(model.add(config_activation_node("1", layer_id)));
                        NANO_CHECK(model.add(config_affine_node("2", cmd_omaps, cmd_orows, cmd_ocols)));
                        NANO_CHECK(model.connect("1", "2"));

                        test_model(model,
                                apsize(cmd_idims, cmd_odims));
                }
        }
}

NANO_CASE(conv3d1)
{
        model_t model;
        NANO_CHECK(model.add(config_conv3d_node("1", 3, 3, 3, 3, 1, 1)));
        NANO_CHECK(model.add(config_activation_node("2", "act-unit")));
        NANO_CHECK(model.add(config_affine_node("3", cmd_omaps, cmd_orows, cmd_ocols)));
        NANO_CHECK(model.connect("1", "2", "3"));

        test_model(model,
                cpsize(cmd_idims, 3, 3, 3, 3) + apsize({3, 4, 4}, cmd_odims));
}

NANO_CASE(conv3d2)
{
        model_t model;
        NANO_CHECK(model.add(config_conv3d_node("1", 4, 3, 3, 1, 1, 1)));
        NANO_CHECK(model.add(config_activation_node("2", "act-snorm")));
        NANO_CHECK(model.add(config_affine_node("3", cmd_omaps, cmd_orows, cmd_ocols)));
        NANO_CHECK(model.connect("1", "2", "3"));

        test_model(model,
                cpsize(cmd_idims, 4, 3, 3, 1) + apsize({4, 4, 4}, cmd_odims));
}

NANO_CASE(conv3d3)
{
        model_t model;
        NANO_CHECK(model.add(config_conv3d_node("1", 5, 3, 3, 1, 1, 1)));
        NANO_CHECK(model.add(config_activation_node("2", "act-splus")));
        NANO_CHECK(model.add(config_affine_node("3", cmd_omaps, cmd_orows, cmd_ocols)));
        NANO_CHECK(model.connect("1", "2", "3"));

        test_model(model,
                cpsize(cmd_idims, 5, 3, 3, 1) + apsize({5, 4, 4}, cmd_odims));
}

NANO_CASE(conv3d4)
{
        model_t model;
        NANO_CHECK(model.add(config_conv3d_node("1", 6, 3, 3, 3, 1, 1)));
        NANO_CHECK(model.add(config_activation_node("2", "act-tanh")));
        NANO_CHECK(model.add(config_affine_node("3", cmd_omaps, cmd_orows, cmd_ocols)));
        NANO_CHECK(model.connect("1", "2", "3"));

        test_model(model,
                cpsize(cmd_idims, 6, 3, 3, 3) + apsize({6, 4, 4}, cmd_odims));
}

NANO_CASE(conv3d_stride1)
{
        model_t model;
        NANO_CHECK(model.add(config_conv3d_node("1", 3, 5, 3, 3, 2, 1)));
        NANO_CHECK(model.add(config_activation_node("2", "act-unit")));
        NANO_CHECK(model.add(config_affine_node("3", cmd_omaps, cmd_orows, cmd_ocols)));
        NANO_CHECK(model.connect("1", "2", "3"));

        test_model(model,
                cpsize(cmd_idims, 3, 5, 3, 3) + apsize({3, 1, 4}, cmd_odims));
}

NANO_CASE(conv3d_stride2)
{
        model_t model;
        NANO_CHECK(model.add(config_conv3d_node("1", 3, 3, 5, 3, 1, 2)));
        NANO_CHECK(model.add(config_activation_node("2", "act-snorm")));
        NANO_CHECK(model.add(config_affine_node("3", cmd_omaps, cmd_orows, cmd_ocols)));
        NANO_CHECK(model.connect("1", "2", "3"));

        test_model(model,
                cpsize(cmd_idims, 3, 3, 5, 3) + apsize({3, 4, 1}, cmd_odims));
}

NANO_CASE(conv3d_stride3)
{
        model_t model;
        NANO_CHECK(model.add(config_conv3d_node("1", 3, 5, 5, 3, 2, 2)));
        NANO_CHECK(model.add(config_activation_node("2", "act-splus")));
        NANO_CHECK(model.add(config_affine_node("3", cmd_omaps, cmd_orows, cmd_ocols)));
        NANO_CHECK(model.connect("1", "2", "3"));

        test_model(model,
                cpsize(cmd_idims, 3, 5, 5, 3) + apsize({3, 1, 1}, cmd_odims));
}

NANO_CASE(norm_global_layer)
{
        model_t model;
        NANO_CHECK(model.add(config_norm3d_node("1", norm_type::global)));
        NANO_CHECK(model.add(config_activation_node("2", "act-snorm")));
        NANO_CHECK(model.add(config_affine_node("3", cmd_omaps, cmd_orows, cmd_ocols)));
        NANO_CHECK(model.connect("1", "2", "3"));

        test_model(model,
                apsize(cmd_idims, cmd_odims));
}

NANO_CASE(norm_plane_layer)
{
        model_t model;
        NANO_CHECK(model.add(config_norm3d_node("1", norm_type::plane)));
        NANO_CHECK(model.add(config_activation_node("2", "act-snorm")));
        NANO_CHECK(model.add(config_affine_node("3", cmd_omaps, cmd_orows, cmd_ocols)));
        NANO_CHECK(model.connect("1", "2", "3"));

        test_model(model,
                apsize(cmd_idims, cmd_odims));
}

NANO_CASE(multi_layer0)
{
        model_t model;
        NANO_CHECK(model.add(config_affine_node("1", 7, 1, 1)));
        NANO_CHECK(model.add(config_activation_node("2", "act-snorm")));
        NANO_CHECK(model.add(config_affine_node("3", 5, 1, 1)));
        NANO_CHECK(model.add(config_activation_node("4", "act-splus")));
        NANO_CHECK(model.add(config_affine_node("5", cmd_omaps, cmd_orows, cmd_ocols)));
        NANO_CHECK(model.connect("1", "2", "3", "4", "5"));

        test_model(model,
                apsize(cmd_idims, {7, 1, 1}) + apsize({7, 1, 1}, {5, 1, 1}) + apsize({5, 1, 1}, cmd_odims));
}

NANO_CASE(multi_layer1)
{
        model_t model;
        NANO_CHECK(model.add(config_conv3d_node("1", 7, 3, 3, 1, 1, 1)));
        NANO_CHECK(model.add(config_activation_node("2", "act-snorm")));
        NANO_CHECK(model.add(config_conv3d_node("3", 4, 1, 1, 1, 1, 1)));
        NANO_CHECK(model.add(config_activation_node("4", "act-splus")));
        NANO_CHECK(model.add(config_affine_node("5", cmd_omaps, cmd_orows, cmd_ocols)));
        NANO_CHECK(model.connect("1", "2", "3", "4", "5"));

        test_model(model,
                cpsize(cmd_idims, 7, 3, 3, 1) + cpsize({7, 4, 4}, 4, 1, 1, 1) + apsize({4, 4, 4}, cmd_odims));
}

NANO_CASE(multi_layer2)
{
        model_t model;
        NANO_CHECK(model.add(config_conv3d_node("1", 7, 3, 3, 1, 1, 1)));
        NANO_CHECK(model.add(config_activation_node("2", "act-snorm")));
        NANO_CHECK(model.add(config_conv3d_node("3", 4, 3, 3, 1, 1, 1)));
        NANO_CHECK(model.add(config_activation_node("4", "act-splus")));
        NANO_CHECK(model.add(config_affine_node("5", cmd_omaps, cmd_orows, cmd_ocols)));
        NANO_CHECK(model.connect("1", "2", "3", "4", "5"));

        test_model(model,
                cpsize(cmd_idims, 7, 3, 3, 1) + cpsize({7, 4, 4}, 4, 3, 3, 1) + apsize({4, 2, 2}, cmd_odims));
}

NANO_CASE(multi_layer3)
{
        model_t model;
        NANO_CHECK(model.add(config_conv3d_node("1", 7, 3, 3, 1, 1, 1)));
        NANO_CHECK(model.add(config_activation_node("2", "act-snorm")));
        NANO_CHECK(model.add(config_conv3d_node("3", 5, 3, 3, 1, 1, 1)));
        NANO_CHECK(model.add(config_activation_node("4", "act-splus")));
        NANO_CHECK(model.add(config_affine_node("5", 5, 1, 1)));
        NANO_CHECK(model.add(config_activation_node("6", "act-splus")));
        NANO_CHECK(model.add(config_affine_node("7", cmd_omaps, cmd_orows, cmd_ocols)));
        NANO_CHECK(model.connect("1", "2", "3", "4", "5", "6", "7"));

        test_model(model,
                cpsize(cmd_idims, 7, 3, 3, 1) + cpsize({7, 4, 4}, 5, 3, 3, 1) + apsize({5, 2, 2}, {5, 1, 1}) + apsize({5, 1, 1}, cmd_odims));
}

NANO_CASE(multi_layer4)
{
        model_t model;
        NANO_CHECK(model.add(config_conv3d_node("1", 8, 3, 3, 1, 1, 1)));
        NANO_CHECK(model.add(config_activation_node("2", "act-snorm")));
        NANO_CHECK(model.add(config_conv3d_node("3", 6, 3, 3, 2, 1, 1)));
        NANO_CHECK(model.add(config_activation_node("4", "act-splus")));
        NANO_CHECK(model.add(config_affine_node("5", 5, 1, 1)));
        NANO_CHECK(model.add(config_activation_node("6", "act-splus")));
        NANO_CHECK(model.add(config_affine_node("7", cmd_omaps, cmd_orows, cmd_ocols)));
        NANO_CHECK(model.connect("1", "2", "3", "4", "5", "6", "7"));

        test_model(model,
                cpsize(cmd_idims, 8, 3, 3, 1) + cpsize({8, 4, 4}, 6, 3, 3, 2) + apsize({6, 2, 2}, {5, 1, 1}) + apsize({5, 1, 1}, cmd_odims));
}

NANO_CASE(multi_layer5)
{
        model_t model;
        NANO_CHECK(model.add(config_conv3d_node("1", 9, 3, 3, 1, 1, 1)));
        NANO_CHECK(model.add(config_activation_node("2", "act-snorm")));
        NANO_CHECK(model.add(config_conv3d_node("3", 6, 3, 3, 3, 1, 1)));
        NANO_CHECK(model.add(config_activation_node("4", "act-splus")));
        NANO_CHECK(model.add(config_affine_node("5", 5, 1, 1)));
        NANO_CHECK(model.add(config_activation_node("6", "act-splus")));
        NANO_CHECK(model.add(config_affine_node("7", cmd_omaps, cmd_orows, cmd_ocols)));
        NANO_CHECK(model.connect("1", "2", "3", "4", "5", "6", "7"));

        test_model(model,
                cpsize(cmd_idims, 9, 3, 3, 1) + cpsize({9, 4, 4}, 6, 3, 3, 3) + apsize({6, 2, 2}, {5, 1, 1}) + apsize({5, 1, 1}, cmd_odims));
}

NANO_CASE(multi_mix_plus4d)
{
        model_t model;
        NANO_CHECK(model.add(config_conv3d_node("11", 4, 5, 5, 1, 1, 1)));
        NANO_CHECK(model.add(config_activation_node("12", "act-snorm")));
        NANO_CHECK(model.add(config_conv3d_node("21", 3, 3, 3, 1, 1, 1)));
        NANO_CHECK(model.add(config_activation_node("22", "act-snorm")));
        NANO_CHECK(model.add(config_conv3d_node("23", 4, 3, 3, 1, 1, 1)));
        NANO_CHECK(model.add(config_activation_node("24", "act-snorm")));
        NANO_CHECK(model.add(config_plus4d_node("xx")));
        NANO_CHECK(model.add(config_activation_node("x1", "act-splus")));
        NANO_CHECK(model.add(config_affine_node("x2", 5, 1, 1)));
        NANO_CHECK(model.add(config_activation_node("x3", "act-splus")));
        NANO_CHECK(model.add(config_affine_node("x4", cmd_omaps, cmd_orows, cmd_ocols)));
        NANO_CHECK(model.connect("11", "12", "xx"));
        NANO_CHECK(model.connect("21", "22", "23", "24", "xx"));
        NANO_CHECK(model.connect("xx", "x1", "x2", "x3", "x4"));

        test_model(model,
                cpsize(cmd_idims, 4, 5, 5, 1) +
                cpsize(cmd_idims, 3, 3, 3, 1) + cpsize({3, 4, 4}, 4, 3, 3, 1) +
                apsize({4, 2, 2}, {5, 1, 1}) + apsize({5, 1, 1}, cmd_odims));
}

NANO_CASE(multi_mix_tcat4d)
{
        model_t model;
        NANO_CHECK(model.add(config_conv3d_node("11", 4, 5, 5, 1, 1, 1)));
        NANO_CHECK(model.add(config_activation_node("12", "act-snorm")));
        NANO_CHECK(model.add(config_conv3d_node("21", 3, 3, 3, 1, 1, 1)));
        NANO_CHECK(model.add(config_activation_node("22", "act-snorm")));
        NANO_CHECK(model.add(config_conv3d_node("23", 4, 3, 3, 1, 1, 1)));
        NANO_CHECK(model.add(config_activation_node("24", "act-snorm")));
        NANO_CHECK(model.add(config_tcat4d_node("xx")));
        NANO_CHECK(model.add(config_activation_node("x1", "act-splus")));
        NANO_CHECK(model.add(config_affine_node("x2", 5, 1, 1)));
        NANO_CHECK(model.add(config_activation_node("x3", "act-splus")));
        NANO_CHECK(model.add(config_affine_node("x4", cmd_omaps, cmd_orows, cmd_ocols)));
        NANO_CHECK(model.connect("11", "12", "xx"));
        NANO_CHECK(model.connect("21", "22", "23", "24", "xx"));
        NANO_CHECK(model.connect("xx", "x1", "x2", "x3", "x4"));

        test_model(model,
                cpsize(cmd_idims, 4, 5, 5, 1) +
                cpsize(cmd_idims, 3, 3, 3, 1) + cpsize({3, 4, 4}, 4, 3, 3, 1) +
                apsize({8, 2, 2}, {5, 1, 1}) + apsize({5, 1, 1}, cmd_odims));
}

NANO_END_MODULE()
