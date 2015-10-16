#pragma once

#include "row.h"
#include "text/from_string.hpp"
#include <algorithm>

namespace ncv
{
        ///
        /// \brief select the column with the minimum value
        ///
        template
        <
                typename tscalar
        >
        decltype(auto) make_table_row_minimum_mark()
        {
                const auto op = text::make_less_from_string<tscalar>();

                return [=] (const table_row_t& row) -> size_t
                {
                        return std::min_element(row.begin(), row.end(), op) - row.begin();
                };
        }

        ///
        /// \brief select the column with the maximum value
        ///
        template
        <
                typename tscalar
        >
        decltype(auto) make_table_row_maximum_mark()
        {
                const auto op = text::make_less_from_string<tscalar>();

                return [=] (const table_row_t& row) -> size_t
                {
                        return std::max_element(row.begin(), row.end(), op) - row.begin();
                };
        }
}

