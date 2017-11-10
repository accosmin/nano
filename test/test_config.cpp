#include "utest.h"
#include "config.h"

using namespace nano;

NANO_BEGIN_MODULE(test_config)

NANO_CASE(construct_pair)
{
        config_t config;

        config.pair(config.none(), "omaps", 128);

        NANO_REQUIRE_NOTHROW(config.get<int>("omaps"));
        NANO_CHECK_EQUAL(config.get<int>("omaps"), 128);

        NANO_CHECK_THROW(config.get<float>("unknown-key"), std::exception);
}

NANO_CASE(construct_pairs)
{
        config_t config;

        config.pair(config.none(), "omaps", 128);
        const auto id_nodes = config.value(config.none(), "nodes");
        config.pairs(
                id_nodes,
                 "name", "conv0", "type", "conv3d", "omaps", 128);

        NANO_REQUIRE_NOTHROW(config.get<int>("nodes", "omaps"));
        NANO_CHECK_EQUAL(config.get<int>("nodes", "omaps"), 128);

        NANO_CHECK_THROW(config.get<string_t>("unknown-key", "another-unknown-key"), std::exception);
}

NANO_CASE(constructor_complex)
{
        config_t config;

        config.pair(config.none(), "version", 4);

        const auto id_nodes = config.value(config.none(), "nodes");
        config.pairs(id_nodes,
                "name", "conv0", "type", "conv3d", "omaps", 128);
        config.pairs(id_nodes,
                "name", "actv0", "type", "act-snorm");
        config.pairs(id_nodes,
                "name", "output", "type", "affine", "omaps", 10, "orows", 1, "ocols", 1);

        // todo: need to specify which element in the array to search
}

NANO_CASE(serialize)
{
        const string_t json = R"XXX(
{
        "protos": [{
                "name": "dconv9x9",
                "type": "conv3d", "omaps": 128, "krows": 9, "kcols": 9, "kconn": 1, "kdrow": 1, "kdcol": 1
        }, {
                "name": "dconv7x7",
                "type": "conv3d", "omaps": 128, "krows": 7, "kcols": 7, "kconn": 1, "kdrow": 1, "kdcol": 1
        }, {
                "name": "dconv5x5",
                "type": "conv3d", "omaps": 128, "krows": 5, "kcols": 5, "kconn": 1, "kdrow": 1, "kdcol": 1
        }, {
                "name": "dconv3x3",
                "type": "conv3d", "omaps": 128, "krows": 3, "kcols": 3, "kconn": 1, "kdrow": 1, "kdcol": 1
        }, {
                "name": "dconv1x1",
                "type": "conv3d", "omaps": 128, "krows": 1, "kcols": 1, "kconn": 1, "kdrow": 1, "kdcol": 1
        }, {
                "name": "act",
                "type": "act-snorm"
        }, {
                "name": "pnorm",
                "type": "norm", "kind": "plane"
        }, {
                "name": "affine1024",
                "type": "affine", "omaps": 1024, "orows": 1, "ocols": 1
        }, {
                "name": "output",
                "type": "affine", "omaps": 10, "orows": 1, "ocols": 1
        }],
        "nodes": [{
                "name": "norm0",        "proto": "pnorm" },

        {       "name": "p11_conv9x9",  "proto": "conv9x9" },
        {       "name": "p12_act,       "proto": "act" },

        {       "name": "p21_conv7x7",  "proto": "conv7x7" },
        {       "name": "p22_act,       "proto": "act" },
        {       "name": "p23_conv3x3",  "proto": "conv3x3" },
        {       "name": "p24_act,       "proto": "act" },

        {       "name": "p31_conv5x5",  "proto": "conv5x5" },
        {       "name": "p32_act,       "proto": "act" },
        {       "name": "p33_conv3x3",  "proto": "conv3x3" },
        {       "name": "p34_act,       "proto": "act" },
        {       "name": "p35_conv3x3",  "proto": "conv3x3" },
        {       "name": "p36_act,       "proto": "act" },

        {       "name": "p41_conv3x3",  "proto": "conv3x3" },
        {       "name": "p42_act,       "proto": "act" },
        {       "name": "p43_conv3x3",  "proto": "conv3x3" },
        {       "name": "p44_act,       "proto": "act" },
        {       "name": "p45_conv3x3",  "proto": "conv3x3" },
        {       "name": "p46_act,       "proto": "act" },
        {       "name": "p47_conv3x3",  "proto": "conv3x3" },
        {       "name": "p48_act,       "proto": "act" },

        {       "name": "affine1",      "proto": "affine1024" },
        {       "name": "act1",         "proto": "act" },
        {       "name": "affine2",      "proto": "affine1024" },
        {       "name": "act2",         "proto": "act" },
        {       "name": "output",       "proto": "output"
        }],
        "model": [
                [ "norm0", "p11_conv9x9", "p12_act", "mix_plus" ],
                [ "norm0", "p21_conv7x7", "p22_act", "p23_conv3x3", "p24_act", "mix_plus" ],
                [ "norm0", "p31_conv5x5", "p32_act", "p33_conv3x3", "p34_act", "p35_conv3x3", "p36_act", "mix_plus" ],
                [ "norm0", "p41_conv3x3", "p42_act", "p43_conv3x3", "p44_act", "p45_conv3x3", "p46_act", "p47_conv3x3", "p48_act", "mix_plus" ],
                [ "mix_plus", "affine1", "act1", "affine2", "act2", "output"]
        ]
}
)XXX";
}

NANO_CASE(deserialize)
{
}

NANO_END_MODULE()
