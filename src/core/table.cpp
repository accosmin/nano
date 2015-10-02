#include "table.h"
#include "text/align.hpp"
#include <cassert>
#include <numeric>
#include <iostream>

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
