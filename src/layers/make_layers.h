#pragma once

#include "text/to_params.h"

namespace nano
{
        ///
        /// \brief utilities to simplify creation of multi-layer models.
        ///
        inline string_t make_layer(const string_t& description)
        {
                return description + ";";
        }

        inline string_t make_affine_layer(const tensor_size_t dims, const string_t& activation = "act-snorm")
        {
                return  make_layer("affine:" + to_params("dims", dims)) +
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

        inline string_t make_conv_layer(
                const tensor_size_t dims, const tensor_size_t rows, const tensor_size_t cols, const tensor_size_t conn,
                const string_t& activation = "act-snorm",
                const tensor_size_t drow = 1, const tensor_size_t dcol = 1)
        {
                return  make_layer("conv:" + to_params("dims", dims, "rows", rows, "cols", cols, "conn", conn, "drow", drow, "dcol", dcol)) +
                        make_layer(activation);
        }
}
