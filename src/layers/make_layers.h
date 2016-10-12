#pragma once

#include "stringi.h"
#include "text/to_params.h"

namespace nano
{
        ///
        /// \brief utilities to simplify creation of multi-layer models for benchmarking and testing.
        ///

        inline string_t make_layer(const string_t& description)
        {
                return description + ";";
        }

        template <typename tsize>
        string_t make_affine_layer(const tsize dims, const string_t& activation = "act-snorm")
        {
                return  make_layer("affine:" + to_params("dims", dims)) +
                        make_layer(activation);
        }

        template <typename tsize>
        string_t make_output_layer(const tsize dims, const string_t& activation = "")
        {
                return  make_affine_layer(dims, activation);
        }

        template <typename tsize>
        string_t make_conv_layer(
                const tsize dims, const tsize rows, const tsize cols, const tsize conn,
                const string_t& activation = "act-snorm", const tsize drow = 1, const tsize dcol = 1)
        {
                return  make_layer("conv:" + to_params(
                                "dims", dims, "rows", rows, "cols", cols, "conn", conn, "drow", drow, "dcol", dcol)) +
                        make_layer(activation);
        }
}
