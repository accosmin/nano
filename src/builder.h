#pragma once

#include "model.h"
#include "core/json.h"
#include "core/algorithm.h"
#include "layers/norm3d_params.h"

namespace nano
{
        ///
        /// \brief check if a computation node is an activation node.
        ///
        inline bool is_activation_node(const string_t& node_id)
        {
                return nano::starts_with(node_id, "act-");
        }

        ///
        /// \brief serialize computation nodes to JSON.
        ///
        template <typename tname>
        json_t config_norm3d_node(const tname& name, const norm_type type)
        {
                return to_json("name", name, "type", norm3d_node_name(),
                        "norm", type);
        }

        template <typename tname>
        json_t config_conv3d_node(const tname& name,
                const tensor_size_t omaps, const tensor_size_t krows, const tensor_size_t kcols,
                const tensor_size_t kconn = 1, const tensor_size_t kdrow = 1, const tensor_size_t kdcol = 1)
        {
                return to_json("name", name, "type", conv3d_node_name(),
                        "omaps", omaps, "krows", krows, "kcols", kcols, "kconn", kconn, "kdrow", kdrow, "kdcol", kdcol);
        }

        template <typename tname>
        json_t config_affine_node(const tname& name,
                const tensor_size_t omaps, const tensor_size_t orows, const tensor_size_t ocols)
        {
                return to_json("name", name, "type", affine_node_name(),
                        "omaps", omaps, "orows", orows, "ocols", ocols);
        }

        template <typename tname, typename ttype>
        json_t config_activation_node(const tname& name, const ttype& type)
        {
                assert(is_activation_node(type));
                return to_json("name", name, "type", type);
        }

        template <typename tname>
        json_t config_plus4d_node(const tname& name)
        {
                return to_json("name", name, "type", plus4d_node_name());
        }

        template <typename tname>
        json_t config_tcat4d_node(const tname& name)
        {
                return to_json("name", name, "type", tcat4d_node_name());
        }
}
