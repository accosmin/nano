#pragma once

#include "row.h"
#include "text/from_string.hpp"
#include <algorithm>

namespace cortex
{
        ///
        /// \brief select the column with the minimum value
        ///
        template
        <
                typename tscalar
        >
        auto make_table_row_minimum_mark()
        {
                const auto op = text::make_less_from_string<tscalar>();

                return [=] (const table_row_t& row) -> size_t
                {
                        return static_cast<size_t>(std::min_element(row.begin(), row.end(), op) - row.begin());
                };
        }

        ///
        /// \brief select the column with the maximum value
        ///
        template
        <
                typename tscalar
        >
        auto make_table_row_maximum_mark()
        {
                const auto op = text::make_less_from_string<tscalar>();

                return [=] (const table_row_t& row) -> size_t
                {
                        return static_cast<size_t>(std::max_element(row.begin(), row.end(), op) - row.begin());
                };
        }
}
