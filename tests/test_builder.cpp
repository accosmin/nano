#include "utest.h"
#include "builder.h"
#include "layers/conv3d_params.h"
#include "layers/affine_params.h"

using namespace nano;

NANO_BEGIN_MODULE(test_builder)

NANO_CASE(affine)
{
        const auto idims = make_dims(4, 13, 11);
        const auto odims = make_dims(3, 11, 12);
        const auto param = affine_params_t{idims, odims};
        NANO_CHECK(param.valid());

        const auto layer = get_layers().get(affine_node_name());
        NANO_REQUIRE(layer);
        layer->from_json(config_affine_node("node", std::get<0>(odims), std::get<1>(odims), std::get<2>(odims)));
        NANO_CHECK(layer->resize({idims}));
        NANO_CHECK_EQUAL(layer->odims(), odims);
        NANO_CHECK_EQUAL(layer->psize(), param.psize());
}

NANO_CASE(conv3d)
{
        const auto idims = make_dims(4, 13, 11);

        for (auto krows = 1; krows <= 3; ++ krows)
        for (auto kcols = 1; kcols <= 3; ++ kcols)
        for (auto kconn = 1; kconn <= 2; ++ kconn)
        for (auto kdrow = 1; kdrow <= 2; ++ kdrow)
        for (auto kdcol = 1; kdcol <= 2; ++ kdcol)
        {
                const auto omaps = 8;
                const auto param = conv3d_params_t{idims, omaps, kconn, krows, kcols, kdrow, kdcol};
                NANO_CHECK(param.valid());

                const auto layer = get_layers().get(conv3d_node_name());
                NANO_REQUIRE(layer);
                layer->from_json(config_conv3d_node("node", omaps, krows, kcols, kconn, kdrow, kdcol));
                NANO_CHECK(layer->resize({idims}));
                NANO_CHECK_EQUAL(layer->odims(), param.odims());
                NANO_CHECK_EQUAL(layer->psize(), param.psize());
        }
}

NANO_CASE(norm3d)
{
        const auto idims = make_dims(4, 13, 11);

        for (auto type : enum_values<norm_type>())
        {
                const auto param = norm3d_params_t{idims, type};
                NANO_CHECK(param.valid());

                const auto layer = get_layers().get(norm3d_node_name());
                NANO_REQUIRE(layer);
                layer->from_json(config_norm3d_node("node", type));
                NANO_CHECK(layer->resize({idims}));
                NANO_CHECK_EQUAL(layer->odims(), idims);
                NANO_CHECK_EQUAL(layer->psize(), 0);
        }
}

NANO_CASE(activation)
{
        const auto idims = make_dims(4, 13, 11);

        for (const auto& node_id : get_layers().ids())
        {
                if (is_activation_node(node_id))
                {
                        const auto layer = get_layers().get(node_id);
                        NANO_REQUIRE(layer);
                        NANO_CHECK(layer->resize({idims}));
                        NANO_CHECK_EQUAL(layer->odims(), idims);
                        NANO_CHECK_EQUAL(layer->psize(), 0);
                }
        }
}

NANO_CASE(mix-plus4d)
{
        const auto idim1 = make_dims(4, 13, 11);
        const auto idim2 = make_dims(4, 13, 11);
        const auto idim3 = make_dims(4, 13, 11);

        const auto layer = get_layers().get(plus4d_node_name());
        NANO_REQUIRE(layer);
        NANO_CHECK(layer->resize({idim1, idim2, idim3}));
        NANO_CHECK_EQUAL(layer->odims(), idim1);
        NANO_CHECK_EQUAL(layer->psize(), 0);
}

NANO_CASE(mix-tcat4d)
{
        const auto idim1 = make_dims(2, 13, 11);
        const auto idim2 = make_dims(3, 13, 11);
        const auto idim3 = make_dims(4, 13, 11);

        const auto layer = get_layers().get(tcat4d_node_name());
        NANO_REQUIRE(layer);
        NANO_CHECK(layer->resize({idim1, idim2, idim3}));
        NANO_CHECK_EQUAL(layer->odims(), make_dims(9, 13, 11));
        NANO_CHECK_EQUAL(layer->psize(), 0);
}

NANO_END_MODULE()
