#pragma once

#include "cortex/string.h"
#include "text/to_string.hpp"

namespace cortex
{
        ///
        /// \brief utilities to simplify layer creation for benchmarking and testing.
        ///

        template <typename tsize>
        string_t make_affine_layer(const tsize dims, const string_t& activation = "act-snorm")
        {
                return "affine:dims=" + text::to_string(dims) + ";" + activation + ";";
        }

        template <typename tsize>
        string_t make_output_layer(const tsize dims)
        {
                return make_affine_layer(dims, "");   // NB: no activation for the output layer!
        }

        template <typename tsize>
        string_t make_conv_pool_layer(const tsize dims, const tsize rows, const tsize cols,
                const string_t& activation = "act-snorm", const string_t& pooling = "pool-max")
        {
                using text::to_string;
                return  "conv:dims=" + to_string(dims) + ",rows=" + to_string(rows) + ",cols=" + to_string(cols) +
                        ";" + activation + ";" + pooling + ";";
        }

        template <typename tsize>
        string_t make_conv_layer(const tsize dims, const tsize rows, const tsize cols,
                const string_t& activation = "act-snorm")
        {
                return make_conv_pool_layer(dims, rows, cols, activation, "");
        }
}
