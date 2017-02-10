#include "align.h"
#include "table.h"
#include <numeric>
#include <ostream>

namespace nano
{
        table_t::table_t(const string_t& title) :
                m_title(title)
        {
                m_rows.reserve(1024);
        }

        const string_t& table_t::title() const
        {
                return m_title;
        }

        table_header_t& table_t::header()
        {
                return m_header;
        }

        const table_header_t& table_t::header() const
        {
                return m_header;
        }

        const table_row_t& table_t::row(const std::size_t index) const
        {
                return m_rows.at(index);
        }

        std::size_t table_t::cols() const
        {
                return header().size();
        }

        std::size_t table_t::rows() const
        {
                return m_rows.size();
        }

        void table_t::clear()
        {
                m_rows.clear();
        }

        std::ostream& operator<<(std::ostream& os, const table_t& table)
        {
                const bool use_row_delim = false;

                // size of the name column (in characters)
                const size_t namesize = [&] ()
                {
                        size_t size = table.title().size();
                        for (size_t r = 0; r < table.rows(); ++ r)
                        {
                                size = std::max(size, table.row(r).name().size());
                        }

                        return size;
                }();

                // size of the value columns (in characters)
                const sizes_t colsizes = [&] ()
                {
                        sizes_t sizes(table.cols(), 0);
                        for (size_t c = 0; c < table.cols(); ++ c)
                        {
                                sizes[c] = std::max(sizes[c], table.header()[c].size());
                        }
                        for (size_t r = 0; r < table.rows(); ++ r)
                        {
                                const auto& row = table.row(r);
                                for (size_t c = 0; c < std::min(table.cols(), row.size()); ++ c)
                                {
                                        sizes[c] = std::max(sizes[c], row.value(c).size() + row.marking(c).size());
                                }
                        }

                        return sizes;
                }();

                // size of the row (in characters)
                const auto rowsize =
                        namesize + 2 +
                        table.cols() * 3 +
                        std::accumulate(colsizes.begin(), colsizes.end(), size_t(0));

                //
                const auto print_row_delim = [&] ()
                {
                        os << "|" << string_t(namesize + 2, '-');
                        for (size_t c = 0; c < table.cols(); ++ c)
                        {
                                os << "|" << string_t(colsizes[c] + 2, '-');
                        }
                        os << "|" << std::endl;
                };

                // display header
                print_row_delim();
                os << nano::align("| " + table.title(), namesize + 3);
                for (size_t c = 0; c < table.cols(); ++ c)
                {
                        os << nano::align("| " + table.header()[c], colsizes[c] + 3);
                }
                os << "|" << std::endl;
                print_row_delim();

                // display rows
                for (size_t r = 0; r < table.rows(); ++ r)
                {
                        const auto& row = table.row(r);

                        if (r > 0 && r < table.rows() && use_row_delim)
                        {
                                os << string_t(rowsize, '-') << std::endl;
                        }

                        os << nano::align("| " + row.name(), namesize + 3);
                        for (size_t c = 0; c < std::min(table.cols(), row.size()); ++ c)
                        {
                                os << nano::align("| " + row.value(c) + row.marking(c), colsizes[c] + 3);
                        }
                        os << "|" << std::endl;
                }
                print_row_delim();

                return os;
        }
}
