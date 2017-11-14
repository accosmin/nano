#pragma once

#include "text/algorithm.h"
#include "text/json_writer.h"

namespace nano
{
        inline string_t json()
        {
                return  R"XXX(
{
"nodes": [{
        "name": "norm0",
        "type": "norm",
        "config": {"type": "plane"}
}, {
        "name": "p11_conv9x9",
        "type": "conv3d",
        "config": {"omaps": 128, "krows": 9, "kcols": 9, "kconn": 1, "kdrow": 1, "kdcol": 1}
}, {
        "name": "p12_act",
        "type": "act-snorm"
}, {
        "name": "p21_conv7x7",
        "type": "conv3d",
        "config": {"omaps": 128, "krows": 7, "kcols": 7, "kconn": 1, "kdrow": 1, "kdcol": 1}
}, {
        "name": "p22_act",
        "type": "act-snorm"
}, {
        "name": "p23_conv3x3",
        "type": "conv3d",
        "config": {"omaps": 128, "krows": 3, "kcols": 3, "kconn": 1, "kdrow": 1, "kdcol": 1}
}, {
        "name": "p24_act",
        "type": "act-snorm"
}, {
        "name": "p31_conv5x5",
        "type": "conv3d",
        "config": {"omaps": 128, "krows": 5, "kcols": 5, "kconn": 1, "kdrow": 1, "kdcol": 1}
}, {
        "name": "p32_act",
        "type": "act-snorm"
}, {
        "name": "p33_conv3x3",
        "type": "conv3d",
        "config": {"omaps": 128, "krows": 3, "kcols": 3, "kconn": 1, "kdrow": 1, "kdcol": 1}
}, {
        "name": "p34_act",
        "type": "act-snorm"
}, {
        "name": "p35_conv3x3",
        "type": "conv3d",
        "config": {"omaps": 128, "krows": 3, "kcols": 3, "kconn": 1, "kdrow": 1, "kdcol": 1}
}, {
        "name": "p36_act",
        "type": "act-snorm"
}, {
        "name": "p41_conv3x3",
        "type": "conv3d",
        "config": {"omaps": 128, "krows": 3, "kcols": 3, "kconn": 1, "kdrow": 1, "kdcol": 1}
}, {
        "name": "p42_act",
        "type": "act-snorm"
}, {
        "name": "p43_conv3x3",
        "type": "conv3d",
        "config": {"omaps": 128, "krows": 3, "kcols": 3, "kconn": 1, "kdrow": 1, "kdcol": 1}
}, {
        "name": "p44_act",
        "type": "act-snorm"
}, {
        "name": "p45_conv3x3",
        "type": "conv3d",
        "config": {"omaps": 128, "krows": 3, "kcols": 3, "kconn": 1, "kdrow": 1, "kdcol": 1}
}, {
        "name": "p46_act",
        "type": "act-snorm"
}, {
        "name": "p47_conv3x3",
        "type": "conv3d",
        "config": {"omaps": 128, "krows": 3, "kcols": 3, "kconn": 1, "kdrow": 1, "kdcol": 1}
}, {
        "name": "p48_act",
        "type": "act-snorm"
}, {
        "name": "mix_plus",
        "type": "plus"
}, {
        "name": "affine1",
        "type": "affine",
        "config": {"omaps": 1024, "orows": 1, "ocols": 1}
}, {
        "name": "act1",
        "type": "act-snorm"
}, {
        "name": "affine2",
        "type": "affine",
        "config": {"omaps": 1024, "orows": 1, "ocols": 1}
}, {
        "name": "act2",
        "type": "act-snorm"
}, {
        "name": "output",
        "type": "affine",
        "config": {"omaps": 10, "orows": 1, "ocols": 1}
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

        ///
        /// \brief check if a computation node is an activation node.
        ///
        inline bool is_activation_node(const string_t& node_id)
        {
                return nano::starts_with(node_id, "act-");
        }

        ///
        /// \brief serialize computation nodes.
        ///
        template <typename tname>
        json_writer_t& add_norm_by_plane_node(json_writer_t& writer, const tname& name)
        {
                return writer.object("name", name, "type", "norm", "kind", "plane");
        }

        template <typename tname>
        json_writer_t& add_norm_globally_node(json_writer_t& writer, const tname& name)
        {
                return writer.object("name", name, "type", "norm", "kind", "global");
        }

        template <typename tname>
        json_writer_t& add_affine_node(json_writer_t& writer, const tname& name,
                const tensor_size_t omaps, const tensor_size_t orows, const tensor_size_t ocols)
        {
                return writer.object("name", name, "type", "affine", "omaps", omaps, "orows", orows, "ocols", ocols);
        }

        template <typename tname>
        json_writer_t& add_affine_node(json_writer_t& writer, const tname& name,
                const tensor3d_dims_t& odims)
        {
                return add_affine_node(writer, name, std::get<0>(odims), std::get<1>(odims), std::get<2>(odims));
        }

        template <typename tname, typename ttype>
        json_writer_t& add_activation_node(json_writer_t& writer, const tname& name, const ttype& type)
        {
                return writer.object("name", name, "type", type);
        }

        template <typename tname>
        json_writer_t& add_conv3d_node(json_writer_t& writer, const tname& name,
                const tensor_size_t omaps, const tensor_size_t krows, const tensor_size_t kcols,
                const tensor_size_t kconn = 1, const tensor_size_t kdrow = 1, const tensor_size_t kdcol = 1)
        {
                return writer.object("name", name, "type", "conv3d",
                        "omaps", omaps, "krows", krows, "kcols", kcols, "kconn", kconn, "kdrow", kdrow, "kdcol", kdcol);
        }

        ///
        /// \brief helper function to serialize a processing path of computation nodes given by their name.
        ///
        template <typename... tnames>
        json_writer_t& connect_nodes(json_writer_t& writer, const tnames&... names)
        {
                return writer.array(names...);
        }
}
