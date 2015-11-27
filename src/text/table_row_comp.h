#pragma once

#include "table_row.h"
#include "from_string.hpp"
#include <cassert>

namespace text
{
        ///
        /// \brief operator to compare two rows
        ///
        template
        <
                typename tindices,
                typename tcomparator
        >
        class table_row_comp_t
        {
        public:

                ///
                /// \brief constructor
                ///
                table_row_comp_t(const tindices& cols, const tcomparator& comp)
                        :       m_cols(cols),
                                m_comp(comp)
                {
                }

                ///
                /// \brief compare
                ///
                bool operator()(const table_row_t& row1, const table_row_t& row2) const
                {
                        assert(!m_cols.empty());
                        assert(row1.size() == row2.size());

                        for (const auto col : m_cols)
                        {
                                assert(col < row1.size());
                                assert(col < row2.size());

                                if (m_comp(row1[col], row2[col]))
                                {
                                        return true;
                                }
                                else if (m_comp(row2[col], row1[col]))
                                {
                                        return false;
                                }
                        }
                        return true;
                }

        private:

                // attributes
                tindices        m_cols;         ///< columns to compare
                tcomparator     m_comp;         ///< operator to compare two scalars
        };

        ///
        /// \brief compare two rows numerically (ascending)
        ///
        template
        <
                typename tscalar,
                typename tindices
        >
        auto make_table_row_ascending_comp(const tindices& cols)
        {
                const auto comp = text::make_less_from_string<tscalar>();

                return table_row_comp_t<tindices, decltype(comp)>(cols, comp);
        }

        ///
        /// \brief compare two rows numerically (descending)
        ///
        template
        <
                typename tscalar,
                typename tindices
        >
        auto make_table_row_descending_comp(const tindices& cols)
        {
                const auto comp = text::make_greater_from_string<tscalar>();

                return table_row_comp_t<tindices, decltype(comp)>(cols, comp);
        }
}

