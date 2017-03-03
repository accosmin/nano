#pragma once

#include "scalar.h"
#include "table_row.h"
#include "from_string.h"
#include <cassert>

namespace nano
{
        namespace detail
        {
                template <typename tscalar, typename toperator>
                auto select_cols(const table_row_t& row, const toperator& op)
                {
                        indices_t indices;
                        for (std::size_t i = 0; i < row.size(); ++ i)
                        {
                                try
                                {
                                        if (op(nano::from_string<tscalar>(row.value(i))))
                                        {
                                                indices.push_back(i);
                                        }
                                }
                                catch (std::exception&) {}
                        }
                        return indices;
                }

                template <typename tscalar>
                auto min_element(const table_row_t& row)
                {
                        const auto op = nano::make_less_from_string<tscalar>();
                        const auto it = std::min_element(row.begin(), row.end(), op);
                        assert(it != row.end());
                        return it;
                }

                template <typename tscalar>
                auto max_element(const table_row_t& row)
                {
                        const auto op = nano::make_less_from_string<tscalar>();
                        const auto it = std::max_element(row.begin(), row.end(), op);
                        assert(it != row.end());
                        return it;
                }
        }

        ///
        /// \brief select the column with the minimum value
        ///
        template <typename tscalar>
        auto make_table_mark_minimum_col()
        {
                return [=] (const table_row_t& row) -> indices_t
                {
                        const auto it = detail::min_element<tscalar>(row);
                        return { static_cast<size_t>(it - row.begin()) };
                };
        }

        ///
        /// \brief select the column with the maximum value
        ///
        template <typename tscalar>
        auto make_table_mark_maximum_col()
        {
                return [=] (const table_row_t& row) -> indices_t
                {
                        const auto it = detail::max_element<tscalar>(row);
                        return { static_cast<size_t>(it - row.begin()) };
                };
        }

        ///
        /// \brief select the columns within [0, epsilon] from the maximum value
        ///
        template <typename tscalar>
        auto make_table_mark_maximum_epsilon_cols(const tscalar epsilon)
        {
                return [=] (const table_row_t& row)
                {
                        const auto it = detail::max_element<tscalar>(row);
                        const auto max = nano::from_string<tscalar>(*it);
                        const auto thres = max - epsilon;

                        return detail::select_cols<tscalar>(row, [thres] (const auto& val) { return val >= thres; });
                };
        }

        ///
        /// \brief select the columns within [0, epsilon] from the minimum value
        ///
        template <typename tscalar>
        auto make_table_mark_minimum_epsilon_cols(const tscalar epsilon)
        {
                return [=] (const table_row_t& row)
                {
                        const auto it = detail::min_element<tscalar>(row);
                        const auto min = nano::from_string<tscalar>(*it);
                        const auto thres = min + epsilon;

                        return detail::select_cols<tscalar>(row, [thres] (const auto& val) { return val <= thres; });
                };
        }

        ///
        /// \brief select the columns within [0, percentage]% from the maximum value
        ///
        template <typename tscalar>
        auto make_table_mark_maximum_percentage_cols(const tscalar percentage)
        {
                return [=] (const table_row_t& row)
                {
                        assert(percentage >= tscalar(1));
                        assert(percentage <= tscalar(99));

                        const auto it = detail::max_element<tscalar>(row);
                        const auto max = nano::from_string<tscalar>(*it);
                        const auto thres = max - percentage * (max < 0 ? -max : +max) / tscalar(100);

                        return detail::select_cols<tscalar>(row, [thres] (const auto& val) { return val >= thres; });
                };
        }

        ///
        /// \brief select the columns within [0, percentage]% from the minimum value
        ///
        template <typename tscalar>
        auto make_table_mark_minimum_percentage_cols(const tscalar percentage)
        {
                return [=] (const table_row_t& row)
                {
                        assert(percentage >= tscalar(1));
                        assert(percentage <= tscalar(99));

                        const auto it = detail::min_element<tscalar>(row);
                        const auto min = nano::from_string<tscalar>(*it);
                        const auto thres = min + percentage * (min < 0 ? -min : +min) / tscalar(100);

                        return detail::select_cols<tscalar>(row, [thres] (const auto& val) { return val <= thres; });
                };
        }
}
