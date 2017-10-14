#include "table.h"
#include <numeric>
#include <fstream>
#include "algorithm.h"
#include "math/numeric.h"

using namespace nano;

cell_t::cell_t() :
        m_span(1), m_alignment(alignment::left)
{
}

cell_t::cell_t(const string_t& data, const size_t span, const alignment align) :
        m_data(data), m_span(span), m_alignment(align)
{
}

row_t::row_t(const mode t) :
        m_type(t),
        m_colspan(1),
        m_alignment(alignment::left)
{
}

size_t row_t::cols() const
{
        return  std::accumulate(m_cells.begin(), m_cells.end(), size_t(0),
                [] (const size_t size, const cell_t& cell) { return size + cell.m_span; });
}

cell_t* row_t::find(const size_t col)
{
        for (size_t icell = 0, icol = 0; icell < m_cells.size(); ++ icell)
        {
                if (icol + m_cells[icell].m_span > col)
                {
                        return &m_cells[icell];
                }
                icol += m_cells[icell].m_span;
        }
        return nullptr;
}

const cell_t* row_t::find(const size_t col) const
{
        for (size_t icell = 0, icol = 0; icell < m_cells.size(); ++ icell)
        {
                if (icol + m_cells[icell].m_span > col)
                {
                        return &m_cells[icell];
                }
                icol += m_cells[icell].m_span;
        }
        return nullptr;
}

void row_t::data(const size_t col, const string_t& str)
{
        cell_t* cell = find(col);
        if (cell)
        {
                cell->data(str);
        }
}

void row_t::mark(const size_t col, const string_t& str)
{
        cell_t* cell = find(col);
        if (cell)
        {
                cell->mark(str);
        }
}

string_t row_t::data(const size_t col) const
{
        const cell_t* cell = find(col);
        return cell ? cell->data() : string_t();
}

string_t row_t::mark(const size_t col) const
{
        const cell_t* cell = find(col);
        return cell ? cell->mark() : string_t();
}

row_t& table_t::header()
{
        m_rows.emplace_back(row_t::mode::header);
        return *m_rows.rbegin();
}

row_t& table_t::append()
{
        m_rows.emplace_back(row_t::mode::data);
        return *m_rows.rbegin();
}

row_t& table_t::delim()
{
        m_rows.emplace_back(row_t::mode::delim);
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
                for (const auto& cell : row.cells())
                {
                        os << cell.m_data;
                        for (size_t i = 0; i < cell.m_span; ++ i)
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

std::ostream& table_t::print(std::ostream& os) const
{
        // size of the value columns (in characters)
        sizes_t colsizes(this->cols(), 0);
        for (const auto& row : m_rows)
        {
                size_t icol = 0;
                for (const auto& cell : row.cells())
                {
                        const auto span = cell.m_span;
                        const auto size = cell.m_data.size() + cell.m_mark.size();
                        for (size_t c = icol; c < icol + span; ++ c)
                        {
                                colsizes[c] = std::max(colsizes[c], idiv(size, span));
                        }
                        icol += span;
                }
        }

        const auto print_row_delim = [&] ()
        {
                for (const auto colsize : colsizes)
                {
                        os << "|" << string_t(colsize + 2, '-');
                }
                os << "|" << std::endl;
        };

        // display rows
        print_row_delim();
        for (const auto& row : m_rows)
        {
                auto it = colsizes.begin();
                switch (row.type())
                {
                case row_t::mode::delim:
                        print_row_delim();
                        break;

                default:
                        for (const auto& cell : row.cells())
                        {
                                const auto colspan = static_cast<std::ptrdiff_t>(cell.m_span);
                                const auto colsize = std::accumulate(it, it + colspan, size_t(0));
                                os << nano::align("| " + cell.data() + cell.mark(), colsize + 3, cell.m_alignment);
                                std::advance(it, colspan);
                        }
                        os << "|" << std::endl;
                        break;
                }
        }
        print_row_delim();

        return os;
}

bool table_t::equals(const table_t& other) const
{
        return std::operator==(m_rows, other.m_rows);
}

std::ostream& nano::operator<<(std::ostream& os, const table_t& table)
{
        return table.print(os);
}

bool nano::operator==(const cell_t& c1, const cell_t& c2)
{
        return  c1.m_data == c2.m_data &&
                c1.m_span == c2.m_span &&
                c1.m_alignment == c2.m_alignment;
}

bool nano::operator==(const row_t& r1, const row_t& r2)
{
        return  r1.type() == r2.type() &&
                std::operator==(r1.cells(), r2.cells());
}

bool nano::operator==(const table_t& t1, const table_t& t2)
{
        return t1.equals(t2);
}
