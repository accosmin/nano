#pragma once

#include "text/config.h"
#include "text/algorithm.h"

namespace nano
{
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
