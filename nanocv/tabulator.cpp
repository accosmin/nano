#include "tabulator.h"
#include <cassert>
#include <numeric>

namespace ncv
{
        tabulator_t::tabulator_t(const string_t& title)
                :       m_title(title)
        {
        }

        tabulator_t::header_t& tabulator_t::header()
        {
                clear();
                return m_header;
        }

        void tabulator_t::clear()
        {
                m_rows.clear();
        }

        tabulator_t::row_t& tabulator_t::append(const string_t& name)
        {
                m_rows.emplace_back(name);
                return *m_rows.rbegin();
        }

        bool tabulator_t::sort(size_t col, const comparator_t& comp)
        {
                if (col < cols())
                {
                        std::stable_sort(m_rows.begin(), m_rows.end(), [&] (const row_t& row1, const row_t& row2)
                        {
                                assert(col < row1.size());
                                assert(row1.size() == cols());

                                assert(col < row2.size());
                                assert(row2.size() == cols());

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

        bool tabulator_t::sort_as_number(size_t col, sorting mode)
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

        std::size_t tabulator_t::border() const
        {
                return 4;
        }

        size_t tabulator_t::name_colsize() const
        {
                size_t colsize = 0;
                for (const row_t& row : m_rows)
                {
                        colsize = std::max(colsize, row.name().size());
                }
                colsize = std::max(colsize, m_title.size()) + border();

                return colsize;
        }

        std::vector<std::size_t> tabulator_t::value_colsizes() const
        {
                std::vector<std::size_t> colsizes(cols(), 0);
                for (size_t c = 0; c < cols(); c ++)
                {
                        colsizes[c] = std::max(colsizes[c], m_header[c].size());
                }
                for (const row_t& row : m_rows)
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

        bool tabulator_t::print(std::ostream& os,
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

                os << text::resize(m_title, ncolsize);
                for (size_t c = 0; c < cols(); c ++)
                {
                        os << text::resize(col_delim + m_header[c], vcolsizes[c]);
                }
                os << std::endl;

                os << string_t(rowsize, row_delim) << std::endl;

                // display rows
                for (size_t r = 0; r < m_rows.size(); r ++)
                {
                        const row_t& row = m_rows[r];

                        if (r > 0 && r < m_rows.size() && use_row_delim)
                        {
                                os << string_t(rowsize, row_delim) << std::endl;
                        }

                        os << text::resize(row.name(), ncolsize);
                        for (size_t c = 0; c < std::min(cols(), row.size()); c ++)
                        {
                                os << text::resize(col_delim + row[c], vcolsizes[c]);
                        }
                        os << std::endl;
                }

                os << string_t(rowsize, table_delim) << std::endl;

                return true;
        }
}
