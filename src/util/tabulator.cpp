#include "tabulator.h"
#include <cassert>

namespace ncv
{
        tabulator_t::tabulator_t(const header_t& header)
                :       m_header(header)
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

        bool tabulator_t::sort(size_t col, comparator_t comp)
        {
                if (col < cols())
                {
                        std::sort(m_rows.begin(), m_rows.end(), [&] (const row_t& row1, const row_t& row2)
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

        bool tabulator_t::print(std::ostream& os) const
        {
                const size_t border = 2;

                // size of name column (in characters)
                size_t name_colsize = 0;
                for (const row_t& row : m_rows)
                {
                        name_colsize = std::max(name_colsize, row.name().size());
                }
                name_colsize += border;

                // size of value columns (in characters)
                size_t value_colsize = 0;
                for (const row_t& row : m_rows)
                {
                        for (const string_t& value : row.values())
                        {
                                value_colsize = std::max(value_colsize, value.size());
                        }
                }
                value_colsize += border;

                // display header
                os << string_t(name_colsize, ' ');
                for (const string_t& colname : m_header.values())
                {
                        os << text::resize(colname, value_colsize);
                }
                os << std::endl;

                // display rows
                for (const row_t& row : m_rows)
                {
                        os << text::resize(row.name(), name_colsize);
                        for (const string_t& value : row.values())
                        {
                                os << text::resize(value, value_colsize);
                        }
                        os << std::endl;
                }

                return true;
        }
}
