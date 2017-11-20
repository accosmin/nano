#pragma once

#include "norm3d_params.h"
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
        /// \brief names for builtin computation nodes.
        ///
        inline const char* conv3d_node_name() { return "conv3d"; }
        inline const char* norm3d_node_name() { return "norm3d"; }
        inline const char* affine_node_name() { return "affine"; }

        ///
        /// \brief configure computation nodes.
        ///
        inline void config_empty_node(json_writer_t& writer)
        {
                writer.object();
        }

        inline void config_norm3d_node(json_writer_t& writer, const norm_type type)
        {
                writer.object("type", type);
        }

        inline void config_conv3d_node(json_writer_t& writer,
                const tensor_size_t omaps, const tensor_size_t krows, const tensor_size_t kcols,
                const tensor_size_t kconn = 1, const tensor_size_t kdrow = 1, const tensor_size_t kdcol = 1)
        {
                writer.object("omaps", omaps, "krows", krows, "kcols", kcols, "kconn", kconn, "kdrow", kdrow, "kdcol", kdcol);
        }

        inline void config_affine_node(json_writer_t& writer,
                const tensor_size_t omaps, const tensor_size_t orows, const tensor_size_t ocols)
        {
                writer.object("omaps", omaps, "orows", orows, "ocols", ocols);
        }

        ///
        /// \brief serialize computation nodes.
        ///
        template <typename tname, typename ttype, typename tconfigurer, typename... targs>
        json_writer_t& add_node(json_writer_t& writer, const tname& name, const ttype& type,
                const tconfigurer& configurer, const targs&... args)
        {
                writer.new_object()
                      .pairs("name", name, "type", type).next()
                      .name("config");
                      configurer(writer, args...);
                return writer.end_object();
        }

        template <typename tname>
        json_writer_t& add_norm3d_node(json_writer_t& writer, const tname& name, const norm_type type)
        {
                return add_node(writer, name, norm3d_node_name(),
                        config_norm3d_node, type);
        }

        template <typename tname>
        json_writer_t& add_conv3d_node(json_writer_t& writer, const tname& name,
                const tensor_size_t omaps, const tensor_size_t krows, const tensor_size_t kcols,
                const tensor_size_t kconn = 1, const tensor_size_t kdrow = 1, const tensor_size_t kdcol = 1)
        {
                return add_node(writer, name, conv3d_node_name(),
                        config_conv3d_node, omaps, krows, kcols, kconn, kdrow, kdcol);
        }

        template <typename tname>
        json_writer_t& add_affine_node(json_writer_t& writer, const tname& name,
                const tensor_size_t omaps, const tensor_size_t orows, const tensor_size_t ocols)
        {
                return add_node(writer, name, affine_node_name(),
                        config_affine_node, omaps, orows, ocols);
        }

        template <typename tname, typename ttype>
        json_writer_t& add_activation_node(json_writer_t& writer, const tname& name, const ttype& type)
        {
                return add_node(writer, name, type,
                        config_empty_node);
        }
}
