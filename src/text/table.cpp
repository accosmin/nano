#include "table.h"
#include "algorithm.h"
#include <numeric>
#include <fstream>

using namespace nano;

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

table_row_t& table_t::append(const table_row_t::storage type)
{
        m_rows.emplace_back(type);
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

        const auto op_header = [=] (const auto& values)
        {
                for (const auto& value : values)
                {
                        header() << value;
                }
        };

        const auto op_append = [=] (const auto& values)
        {
                if (values.size() != cols())
                {
                        return false;
                }
                else
                {
                        auto& row = append();
                        for (const auto& value : values)
                        {
                                row << value;
                        }
                        return true;
                }
        };

        size_t count = 0;
        for (string_t line; std::getline(is, line); ++ count)
        {
                const auto tokens = nano::split(line, delim.c_str());
                if (!tokens.empty() && !line.empty())
                {
                        if (!count && load_header)
                        {
                                op_header(tokens);
                        }
                        else if (!op_append(tokens))
                        {
                                return false;
                        }
                }
        }

        return is.eof();
}

std::ostream& nano::operator<<(std::ostream& os, const table_t& table)
{
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

                switch (row.type())
                {
                case table_row_t::storage::delim:
                        print_row_delim();
                        break;

                default:
                        for (size_t c = 0; c < std::min(table.cols(), row.size()); ++ c)
                        {
                                os << nano::align("| " + row.value(c) + row.marking(c), colsizes[c] + 3);
                        }
                        os << "|" << std::endl;
                        break;
                }
        }
        print_row_delim();

        return os;
}

bool nano::operator==(const table_t& t1, const table_t& t2)
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
