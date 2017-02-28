#include "align.h"
#include "table.h"
#include <numeric>
#include <fstream>

namespace nano
{
        table_t::table_t()
        {
                m_rows.reserve(1024);
        }

        table_header_t& table_t::header()
        {
                return m_header;
        }

        const table_header_t& table_t::header() const
        {
                return m_header;
        }

        table_row_t& table_t::append()
        {
                m_rows.push_back(table_row_t());
                return *m_rows.rbegin();
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

                for (size_t c = 0; c < cols(); ++ c)
                {
                        os << m_header.value(c) << (c + 1 == cols() ? "" : delim);
                }
                os << std::endl;

                for (const auto& row : m_rows)
                {
                        for (size_t c = 0; c < row.size(); ++ c)
                        {
                                os << row.value(c) << (c + 1 == row.size() ? "" : delim);
                        }
                        os << std::endl;
                }

                return true;
        }

        bool table_t::load(const string_t& path, const string_t& delim, const bool load_header)
        {
                std::ifstream is(path.c_str());
                if (!is.is_open())
                {
                        return false;
                }

                if (load_header)
                {
                        m_header = table_header_t();
                }
                clear();

                const auto op_header = [=] (const auto& tokens)
                {
                        for (const auto& token : tokens)
                        {
                                header() << token;
                        }
                };

                const auto op_append = [=] (const auto& tokens)
                {
                        if (tokens.size() != cols())
                        {
                                return false;
                        }

                        auto& row = append();
                        for (const auto& token : tokens)
                        {
                                row << token;
                        }

                        return true;
                };

                size_t count = 0;
                for (string_t line; std::getline(is, line); ++ count)
                {
                        const auto tokens = nano::split(line, delim.c_str());
                        if (tokens.empty() || line.empty())
                        {
                                continue;
                        }

                        if (!count && load_header)
                        {
                                op_header(tokens);
                        }
                        else
                        {
                                if (!op_append(tokens))
                                {
                                        return false;
                                }
                        }
                }

                return is.eof();
        }

        std::ostream& operator<<(std::ostream& os, const table_t& table)
        {
                const bool use_row_delim = false;

                // size of the value columns (in characters)
                const sizes_t colsizes = [&] ()
                {
                        sizes_t sizes(table.cols(), 0);
                        for (size_t c = 0; c < table.cols(); ++ c)
                        {
                                sizes[c] = std::max(sizes[c], table.header().value(c).size());
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
                        table.cols() * 3 +
                        std::accumulate(colsizes.begin(), colsizes.end(), size_t(0));

                //
                const auto print_row_delim = [&] ()
                {
                        for (size_t c = 0; c < table.cols(); ++ c)
                        {
                                os << "|" << string_t(colsizes[c] + 2, '-');
                        }
                        os << "|" << std::endl;
                };

                // display header
                print_row_delim();
                for (size_t c = 0; c < table.cols(); ++ c)
                {
                        os << nano::align("| " + table.header().value(c), colsizes[c] + 3);
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

                return  t1.header() == t2.header() &&
                        t1.rows() == t2.rows() &&
                        rows_equal();
        }
}
