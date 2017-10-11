#include "table.h"
#include "algorithm.h"
#include <numeric>
#include <fstream>

using namespace nano;

cell_t::cell_t() :
        m_span(1), m_align(alignment::left)
{
}

void cell_t::print(std::ostream& os, const size_t maximum) const
{
        os << nano::align(m_data, maximum, m_align);
}

row_t::row_t(const type t) :
        m_type(t)
{
}

row_t& row_t::operator<<(const cell_t& cell)
{
        m_cells.push_back(cell);
        return *this;
}

size_t row_t::cols() const
{
        return  std::accumulate(m_cells.begin(), m_cells.end(), size_t(0),
                [] (const size_t size, const cell_t& cell) { return size + cell.m_span; });
}

cell_t row_t::find(const size_t column) const
{
        for (size_t i = 0, colstart = 0; i < m_cells.size(); ++ i)
        {
                if (column >= colstart)
                {
                        return m_cells[i];
                }
                colstart += m_cells[i].m_span;
        }
        return cell_t{};
}

row_t& table_t::header()
{
        m_rows.emplace_back(row_t::type::header);
        return *m_rows.rbegin();
}

row_t& table_t::append()
{
        m_rows.emplace_back(row_t::type::data);
        return *m_rows.rbegin();
}

row_t& table_t::delim()
{
        m_rows.emplace_back(row_t::type::delim);
        return *m_rows.rbegin();
}

std::size_t table_t::cols() const
{
        const auto op = [] (const row_t& row1, const row_t& row2) { return row1.cols() < row2.cols(); };
        const auto it = std::max_element(m_rows.begin(), m_rows.end(), op);
        return (it == m_rows.end()) ? size_t(0) : it->cols();
}

std::size_t table_t::rows() const
{
        return m_rows.size();
}

bool table_t::save(const string_t& path, const string_t& delim) const
{
        std::ofstream os(path.c_str(), std::ios::trunc);
        if (!os.is_open())
        {
                return false;
        }

        for (const auto& row : m_rows)
        {
                for (const auto& cell : row.m_cells)
                {
                        os << cell.m_data;
                        for (size_t i = 0; i + 1 < cell.m_span; ++ i)
                        {
                                os << delim;
                        }
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

        m_rows.clear();

        const auto op_row = [&] (auto& row, const auto& values)
        {
                for (const auto& value : values)
                {
                        row << value;
                }
        };

        const auto op_header = [=] (const auto& values)
        {
                op_row(header(), values);
        };

        const auto op_append = [=] (const auto& values)
        {
                if (values.size() != cols())
                {
                        return false;
                }
                else
                {
                        op_row(append(), values);
                        return true;
                }
        };

        // todo: this does not handle missing values
        // todo: this does not handle delimiting rows

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

bool nano::operator==(const cell_t& c1, const cell_t& c2)
{
        return  c1.m_data == c2.m_data &&
                c1.m_span == c2.m_span &&
                c1.m_align = c2.m_align;
}

bool nano::operator==(const row_t& r1, const row_t& r2)
{
        return std::operator==(r1.m_cells, r2.m_cells);
}

bool nano::operator==(const table_t& t1, const table_t& t2)
{
        return std::operator==(t1.m_rows, t2.m_rows);
}
