#include "table.h"
#include "text/align.hpp"
#include "text/from_string.hpp"
#include <cassert>
#include <numeric>
#include <iostream>
#include <algorithm>

namespace ncv
{
        table_t::table_t(const string_t& title)
                :       m_title(title)
        {
        }

        table_header_t& table_t::header()
        {
                clear();
                return m_header;
        }

        void table_t::clear()
        {
                m_rows.clear();
        }

        table_row_t& table_t::append(const string_t& name)
        {
                m_rows.emplace_back(name);
                return *m_rows.rbegin();
        }

        bool table_t::sort(size_t col, const comparator_t& comp)
        {
                if (col < cols())
                {
                        std::stable_sort(m_rows.begin(), m_rows.end(), [&] (const auto& row1, const auto& row2)
                        {
                                assert(col < row1.size());
                                assert(row1.size() == this->cols());

                                assert(col < row2.size());
                                assert(row2.size() == this->cols());

                                return comp(row1.values()[col], row2.values()[col]);
                        });

                        return true;
                }

                else
                {
                        // invalid index
                        return false;
                }
        }

        bool table_t::sort_as_number(size_t col, sorting mode)
        {
                switch (mode)
                {
                case sorting::ascending:
                        return sort(col, [] (const string_t& value1, const string_t& value2)
                        {
                                return text::from_string<double>(value1) < text::from_string<double>(value2);
                        });

                case sorting::descending:
                        return sort(col, [] (const string_t& value1, const string_t& value2)
                        {
                                return text::from_string<double>(value1) > text::from_string<double>(value2);
                        });

                default:
                        return false;
                }
        }

        bool table_t::mark(const marker_t& op, const char* marker_string)
        {
                for (auto& row : m_rows)
                {
                        const auto col = op(row.values());
                        if (col < cols())
                        {
                                row[col] += marker_string;
                        }
                }

                return true;
        }

        namespace
        {
                const auto op_comp = [] (const string_t& value1, const string_t& value2)
                {
                        return text::from_string<double>(value1) < text::from_string<double>(value2);
                };

                const auto op_marker = [] (const strings_t& values)
                {
                        return std::min_element(values.begin(), values.end(), op_comp) - values.begin();
                };
        }

        bool table_t::mark_min_number(const char* marker_string)
        {
                return mark(op_marker, marker_string);
        }

        std::size_t table_t::border() const
        {
                return 4;
        }

        size_t table_t::name_colsize() const
        {
                size_t colsize = 0;
                for (const auto& row : m_rows)
                {
                        colsize = std::max(colsize, row.name().size());
                }
                colsize = std::max(colsize, m_title.size()) + border();

                return colsize;
        }

        std::vector<std::size_t> table_t::value_colsizes() const
        {
                std::vector<std::size_t> colsizes(cols(), 0);
                for (size_t c = 0; c < cols(); c ++)
                {
                        colsizes[c] = std::max(colsizes[c], m_header[c].size());
                }
                for (const auto& row : m_rows)
                {
                        for (size_t c = 0; c < std::min(cols(), row.size()); c ++)
                        {
                                colsizes[c] = std::max(colsizes[c], row[c].size());
                        }
                }
                for (size_t& colsize : colsizes)
                {
                        colsize += border();
                }

                return colsizes;
        }

        bool table_t::print(std::ostream& os,
                const char table_delim,
                const char row_delim, bool use_row_delim) const
        {
                const char col_delim = ' ';

                // size of name & value columns (in characters)
                const auto ncolsize = this->name_colsize();
                const auto vcolsizes = this->value_colsizes();

                const auto rowsize = ncolsize + std::accumulate(vcolsizes.begin(), vcolsizes.end(), size_t(0));

                // display header
                os << string_t(rowsize, table_delim) << std::endl;

                os << text::align(m_title, ncolsize);
                for (size_t c = 0; c < cols(); c ++)
                {
                        os << text::align(col_delim + m_header[c], vcolsizes[c]);
                }
                os << std::endl;

                os << string_t(rowsize, row_delim) << std::endl;

                // display rows
                for (size_t r = 0; r < m_rows.size(); r ++)
                {
                        const auto& row = m_rows[r];

                        if (r > 0 && r < m_rows.size() && use_row_delim)
                        {
                                os << string_t(rowsize, row_delim) << std::endl;
                        }

                        os << text::align(row.name(), ncolsize);
                        for (size_t c = 0; c < std::min(cols(), row.size()); c ++)
                        {
                                os << text::align(col_delim + row[c], vcolsizes[c]);
                        }
                        os << std::endl;
                }

                os << string_t(rowsize, table_delim) << std::endl;

                return true;
        }
}
