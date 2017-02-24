#include "align.h"
#include "table.h"
#include <numeric>
#include <fstream>

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

        bool table_t::save(const string_t& path, const string_t& delim) const
        {
                std::ofstream os(path.c_str(), std::ios::trunc);
                if (!os.is_open())
                {
                        return false;
                }

                os << m_title;
                for (const auto& hname : m_header.values())
                {
                        os << delim << hname;
                }
                os << std::endl;

                for (const auto& row : m_rows)
                {
                        os << row.name();
                        for (const auto& value : row.values())
                        {
                                os << delim << value;
                        }
                        os << std::endl;
                }

                return true;
        }

        bool table_t::load(const string_t& path, const string_t& delim)
        {
                std::ifstream is(path.c_str());
                if (!is.is_open())
                {
                        return false;
                }

                m_header = table_header_t();
                clear();

                const auto op_header = [=] (const auto& tokens)
                {
                        m_title = tokens[0];
                        for (size_t i = 1; i < tokens.size(); ++ i)
                        {
                                header() << tokens[i];
                        }
                };

                const auto op_append = [=] (const auto& tokens)
                {
                        auto& row = append(tokens[0]);
                        for (size_t i = 1; i < tokens.size(); ++ i)
                        {
                                row << tokens[i];
                        }
                };

                size_t count = 0;
                for (string_t line; std::getline(is, line); ++ count)
                {
                        const auto tokens = nano::split(line, delim.c_str());
                        if (tokens.empty())
                        {
                                return false;
                        }

                        switch (count)
                        {
                        case 0:         op_header(tokens); break;
                        default:        op_append(tokens); break;
                        }
                }

                return is.eof();
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

        bool operator==(const table_t& t1, const table_t& t2)
        {
                const auto rows_equal = [&] ()
                {
                        for (std::size_t i = 0; i < t1.rows(); ++ i)
                        {
                                if (t1.row(i) != t2.row(i))
                                {
                                        return false;
                                }
                        }
                        return true;
                };

                return  t1.title() == t2.title() &&
                        t1.header() == t2.header() &&
                        t1.rows() == t2.rows() &&
                        rows_equal();
        }
}
