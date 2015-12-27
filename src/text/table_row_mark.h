#pragma once

#include "table_row.h"
#include "from_string.hpp"
#include <algorithm>

namespace text
{
        ///
        /// \brief select the column with the minimum value
        ///
        template
        <
                typename tscalar
        >
        auto make_table_mark_minimum_col()
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
        auto make_table_mark_maximum_col()
        {
                const auto op = text::make_less_from_string<tscalar>();

                return [=] (const table_row_t& row) -> size_t
                {
                        return static_cast<size_t>(std::max_element(row.begin(), row.end(), op) - row.begin());
                };
        }
}

