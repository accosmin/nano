#include "layer.h"
#include "utest.h"
#include "layers/make_layers.h"
#include "layers/conv_params.h"
#include "layers/affine_params.h"

using namespace nano;

NANO_BEGIN_MODULE(test_build)

NANO_CASE(affine)
{
        const auto idims = make_dims(4, 13, 11);
        const auto odims = make_dims(3, 11, 12);
        const auto lname = "name";
        const auto param = affine_params_t{idims, odims};
        NANO_CHECK(param.valid());

        const auto layer = get_layers().get("affine", make_affine_layer(odims));
        NANO_CHECK(layer->configure(idims, lname));
        NANO_CHECK_EQUAL(layer->idims(), idims);
        NANO_CHECK_EQUAL(layer->odims(), odims);
        NANO_CHECK_EQUAL(layer->psize(), param.psize());
}

NANO_CASE(conv3d)
{
        const auto idims = make_dims(4, 13, 11);
        const auto lname = "name";

        for (auto krows = 1; krows <= 3; ++ krows)
        for (auto kcols = 1; kcols <= 3; ++ kcols)
        for (auto kconn = 1; kconn <= 2; ++ kconn)
        for (auto kdrow = 1; kdrow <= 2; ++ kdrow)
        for (auto kdcol = 1; kdcol <= 2; ++ kdcol)
        {
                const auto omaps = 8;
                const auto param = conv_params_t{idims, omaps, kconn, krows, kcols, kdrow, kdcol};
                NANO_CHECK(param.valid());

                const auto layer = get_layers().get("conv3d", make_conv3d_layer(omaps, krows, kcols, kconn, "", kdrow, kdcol));
                NANO_CHECK(layer->configure(idims, lname));
                NANO_CHECK_EQUAL(layer->idims(), idims);
                NANO_CHECK_EQUAL(layer->odims(), param.odims());
                NANO_CHECK_EQUAL(layer->psize(), param.psize());
        }
}

NANO_CASE(norm_by_plane)
{
        const auto idims = make_dims(4, 13, 11);
        const auto lname = "name";

        const auto layer = get_layers().get("norm", make_norm_by_plane_layer());
        NANO_CHECK(layer->configure(idims, lname));
        NANO_CHECK_EQUAL(layer->idims(), idims);
        NANO_CHECK_EQUAL(layer->odims(), idims);
        NANO_CHECK_EQUAL(layer->psize(), 0);
}

NANO_CASE(norm_globally)
{
        const auto idims = make_dims(4, 13, 11);
        const auto lname = "name";

        const auto layer = get_layers().get("norm", make_norm_globally_layer());
        NANO_CHECK(layer->configure(idims, lname));
        NANO_CHECK_EQUAL(layer->idims(), idims);
        NANO_CHECK_EQUAL(layer->odims(), idims);
        NANO_CHECK_EQUAL(layer->psize(), 0);
}

NANO_CASE(activation)
{
        const auto idims = make_dims(4, 13, 11);
        const auto lname = "name";

        for (const auto& layer_id : get_layers().ids())
        {
                if (is_activation_layer(layer_id))
                {
                        const auto layer = get_layers().get(layer_id);
                        NANO_CHECK(layer->configure(idims, lname));
                        NANO_CHECK_EQUAL(layer->idims(), idims);
                        NANO_CHECK_EQUAL(layer->odims(), idims);
                        NANO_CHECK_EQUAL(layer->psize(), 0);
                }
        }
}

NANO_END_MODULE()
