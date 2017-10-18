#pragma once

#include "text/config.h"

namespace nano
{
        ///
        /// \brief utilities to simplify creation of multi-layer models.
        ///
        inline string_t make_layer(const string_t& description)
        {
                return description.empty() ? string_t() : (description + ";");
        }

        inline string_t make_affine_layer(const tensor_size_t omaps, const string_t& activation = "act-snorm")
        {
                return  make_layer("affine:" + to_params("omaps", omaps)) +
                        make_layer(activation);
        }

        inline string_t make_output_layer(const tensor_size_t dims, const string_t& activation = "")
        {
                return  make_affine_layer(dims, activation);
        }

        inline string_t make_output_layer(const tensor3d_dims_t& dims, const string_t& activation = "")
        {
                return make_output_layer(nano::size(dims), activation);
        }

        inline string_t make_conv3d_layer(
                const tensor_size_t omaps, const tensor_size_t krows, const tensor_size_t kcols, const tensor_size_t kconn,
                const string_t& activation = "act-snorm", const tensor_size_t kdrow = 1, const tensor_size_t kdcol = 1)
        {
                return  make_layer("conv:" + to_params("omaps", omaps, "krows", krows, "kcols", kcols, "kconn", kconn, "kdrow", kdrow, "kdcol", kdcol)) +
                        make_layer(activation);
        }
}
