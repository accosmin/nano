#include "table.h"
#include "text/align.hpp"
#include <cassert>
#include <numeric>
#include <iostream>

namespace cortex
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

        std::size_t table_t::cols() const
        {
                return m_header.size();
        }

        std::size_t table_t::rows() const
        {
                return m_rows.size();
        }

        size_t table_t::name_colsize() const
        {
                size_t colsize = m_title.size();
                for (const auto& row : m_rows)
                {
                        colsize = std::max(colsize, row.name().size());
                }

                return colsize;
        }

        std::vector<std::size_t> table_t::value_colsizes() const
        {
                std::vector<std::size_t> colsizes(cols(), 0);

                for (size_t c = 0; c < cols(); ++ c)
                {
                        colsizes[c] = std::max(colsizes[c], m_header[c].size());
                }

                for (const auto& row : m_rows)
                {
                        assert(cols() == row.size());
                        for (size_t c = 0; c < cols(); ++ c)
                        {
                                colsizes[c] = std::max(colsizes[c], row[c].size());
                        }
                }

                return colsizes;
        }

        void table_t::print_row_delim(std::ostream& os) const
        {
                const auto namesize = name_colsize();
                const auto colsizes = value_colsizes();

                os << "|" << string_t(namesize + 2, '-');
                for (size_t c = 0; c < cols(); ++ c)
                {
                        os << "+" << string_t(colsizes[c] + 2, '-');
                }
                os << "|" << std::endl;

//                os << "|" + string_t(rowsize, '-') << "|" << std::endl;
        }

        void table_t::print(std::ostream& os, const bool use_row_delim) const
        {
                // size of name & value columns (in characters)
                const auto namesize = name_colsize();
                const auto colsizes = value_colsizes();

                const auto rowsize =
                        namesize + 2 +
                        cols() * 3 +
                        std::accumulate(colsizes.begin(), colsizes.end(), size_t(0));

                // display header
                print_row_delim(os);

                os << text::align("| " + m_title, namesize + 3);
                for (size_t c = 0; c < cols(); ++ c)
                {
                        os << text::align("| " + m_header[c], colsizes[c] + 3);
                }
                os << "|" << std::endl;

                print_row_delim(os);

                // display rows
                for (size_t r = 0; r < m_rows.size(); ++ r)
                {
                        const auto& row = m_rows[r];

                        if (r > 0 && r < m_rows.size() && use_row_delim)
                        {
                                os << string_t(rowsize, '-') << std::endl;
                        }

                        os << text::align("| " + row.name(), namesize + 3);
                        for (size_t c = 0; c < cols(); ++ c)
                        {
                                os << text::align("| " + row[c], colsizes[c] + 3);
                        }
                        os << "|" << std::endl;
                }

                print_row_delim(os);
        }
}
