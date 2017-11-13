#pragma once

#include "text/config.h"
#include "text/algorithm.h"

namespace nano
{
        inline string_t json()
        {
                return  R"XXX(
{
        "nodes": [{
                "name": "norm0",
                "type": "norm", "kind": "plane"
        }, {
                "name": "p11_conv9x9",
                "type": "conv3d", "omaps": 128, "krows": 9, "kcols": 9, "kconn": 1, "kdrow": 1, "kdcol": 1
        }, {
                "name": "p12_act",
                "type": "act-snorm"
        }, {
                "name": "p21_conv7x7",
                "type": "conv3d", "omaps": 128, "krows": 7, "kcols": 7, "kconn": 1, "kdrow": 1, "kdcol": 1
        }, {
                "name": "p22_act",
                "type": "act-snorm"
        }, {
                "name": "p23_conv3x3",
                "type": "conv3d", "omaps": 128, "krows": 3, "kcols": 3, "kconn": 1, "kdrow": 1, "kdcol": 1
        }, {
                "name": "p24_act",
                "type": "act-snorm"
        }, {
                "name": "p31_conv5x5",
                "type": "conv3d", "omaps": 128, "krows": 5, "kcols": 5, "kconn": 1, "kdrow": 1, "kdcol": 1
        }, {
                "name": "p32_act",
                "type": "act-snorm"
        }, {
                "name": "p33_conv3x3",
                "type": "conv3d", "omaps": 128, "krows": 3, "kcols": 3, "kconn": 1, "kdrow": 1, "kdcol": 1
        }, {
                "name": "p34_act",
                "type": "act-snorm"
        }, {
                "name": "p35_conv3x3",
                "type": "conv3d", "omaps": 128, "krows": 3, "kcols": 3, "kconn": 1, "kdrow": 1, "kdcol": 1
        }, {
                "name": "p36_act",
                "type": "act-snorm"
        }, {
                "name": "p41_conv3x3",
                "type": "conv3d", "omaps": 128, "krows": 3, "kcols": 3, "kconn": 1, "kdrow": 1, "kdcol": 1
        }, {
                "name": "p42_act",
                "type": "act-snorm"
        }, {
                "name": "p43_conv3x3",
                "type": "conv3d", "omaps": 128, "krows": 3, "kcols": 3, "kconn": 1, "kdrow": 1, "kdcol": 1
        }, {
                "name": "p44_act",
                "type": "act-snorm"
        }, {
                "name": "p45_conv3x3",
                "type": "conv3d", "omaps": 128, "krows": 3, "kcols": 3, "kconn": 1, "kdrow": 1, "kdcol": 1
        }, {
                "name": "p46_act",
                "type": "act-snorm"
        }, {
                "name": "p47_conv3x3",
                "type": "conv3d", "omaps": 128, "krows": 3, "kcols": 3, "kconn": 1, "kdrow": 1, "kdcol": 1
        }, {
                "name": "p48_act",
                "type": "act-snorm"
        }, {
                "name": "mix_plus",
                "type": "plus"
        }, {
                "name": "affine1",
                "type": "affine", "omaps": 1024, "orows": 1, "ocols": 1
        }, {
                "name": "act1",
                "type": "act-snorm"
        }, {
                "name": "affine2",
                "type": "affine", "omaps": 1024, "orows": 1, "ocols": 1
        }, {
                "name": "act2",
                "type": "act-snorm"
        }, {
                "name": "output",
                "type": "affine", "omaps": 10, "orows": 1, "ocols": 1
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

        inline bool is_activation_layer(const string_t& layer_id)
        {
                return nano::starts_with(layer_id, "act-");
        }

        inline string_t make_layer(const string_t& config) { return config + ";"; }
        inline string_t make_layer(const string_t& id, const string_t& params) { return make_layer(id + ":" + params); }

        ///
        /// \brief construct normalization layers.
        ///
        inline string_t make_norm_by_plane_layer() { return make_layer("norm", "type=plane"); }
        inline string_t make_norm_globally_layer() { return make_layer("norm", "type=global"); }

        ///
        /// \brief construct affine layers.
        ///
        inline string_t make_affine_layer(const string_t& params, const string_t& activation = "act-snorm")
        {
                return make_layer("affine", params) + make_layer(activation);
        }

        inline string_t make_affine_layer(
                const tensor_size_t omaps, const tensor_size_t orows, const tensor_size_t ocols,
                const string_t& activation = "act-snorm")
        {
                return make_affine_layer(
                       to_params("omaps", omaps, "orows", orows, "ocols", ocols),
                       activation);
        }

        inline string_t make_affine_layer(const tensor3d_dims_t& odims, const string_t& activation = "act-snorm")
        {
                return make_affine_layer(std::get<0>(odims), std::get<1>(odims), std::get<2>(odims), activation);
        }

        ///
        /// \brief construct output (affine) layers.
        ///
        inline string_t make_output_layer(
                const tensor_size_t omaps, const tensor_size_t orows, const tensor_size_t ocols,
                const string_t& activation = string_t())
        {
                return make_affine_layer(omaps, orows, ocols, activation);
        }

        inline string_t make_output_layer(const tensor3d_dims_t& odims, const string_t& activation = string_t())
        {
                return make_output_layer(std::get<0>(odims), std::get<1>(odims), std::get<2>(odims), activation);
        }

        ///
        /// \brief construct 3D convolution layers.
        ///
        inline string_t make_conv3d_layer(const string_t& params, const string_t& activation = "act-snorm")
        {
                return make_layer("conv3d", params) + make_layer(activation);
        }

        inline string_t make_conv3d_layer(
                const tensor_size_t omaps, const tensor_size_t krows, const tensor_size_t kcols, const tensor_size_t kconn,
                const string_t& activation = "act-snorm", const tensor_size_t kdrow = 1, const tensor_size_t kdcol = 1)
        {
                return  make_conv3d_layer(
                        to_params("omaps", omaps, "krows", krows, "kcols", kcols, "kconn", kconn, "kdrow", kdrow, "kdcol", kdcol),
                        activation);
        }
}
