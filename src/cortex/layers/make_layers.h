#pragma once

#include "stringi.h"
#include "text/to_string.hpp"

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
                return  make_layer("affine:dims=" + to_string(dims)) +
                        make_layer(activation);
        }

        template <typename tsize>
        string_t make_output_layer(const tsize dims)
        {
                return make_affine_layer(dims, "");
        }

        template <typename tsize>
        string_t make_conv_pool_layer(const tsize dims, const tsize rows, const tsize cols, const tsize conn,
                const string_t& activation = "act-snorm", const string_t& pooling = "pool-max")
        {
                return  make_layer(
                        "conv:dims=" + to_string(dims) + ",rows=" + to_string(rows) +
                        ",cols=" + to_string(cols) + ",conn=" + to_string(conn)) +
                        make_layer(activation) +
                        make_layer(pooling);
        }

        template <typename tsize>
        string_t make_conv_layer(const tsize dims, const tsize rows, const tsize cols, const tsize conn,
                const string_t& activation = "act-snorm")
        {
                return make_conv_pool_layer(dims, rows, cols, conn, activation, "");
        }
}
