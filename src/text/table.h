#pragma once

#include "arch.h"
#include "scalar.h"
#include <algorithm>
#include "algorithm.h"
#include "to_string.h"
#include "from_string.h"

namespace nano
{
        struct table_t;

        struct cell_t
        {
                cell_t();
                cell_t(const string_t& data, const size_t span, const alignment);

                bool empty() const { return m_data.empty(); }
                void print(std::ostream&, const size_t maximum) const;

                // attributes
                string_t                m_data;
                string_t                m_mark;
                size_t                  m_span;
                alignment               m_alignment;
        };

        struct row_t
        {
                enum class type
                {
                        data,
                        delim,
                        header,
                };

                row_t(const type t = type::data);

                template <typename tvalue>
                row_t& operator<<(const tvalue value)
                {
                        m_cells.emplace_back(to_string(value), colspan(), align());
                        return *this;
                }

                size_t cols() const;
                void mark(size_t col, const string_t& marker);

                const auto& cells() const { return m_cells; }
                const auto& cell(const size_t c) const { return m_cells.at(c); }

                size_t colspan() const { return m_colspan; }
                alignment align() const { return m_alignment; }
                row_t& colspan(const size_t span) { m_colspan = span; return *this; }
                row_t& align(const alignment align) { m_alignment = align; return *this; }

        private:

                // attributes
                type                    m_type;
                size_t                  m_colspan;      ///< current column span
                alignment               m_alignment;    ///< current cell alignment
                std::vector<cell_t>     m_cells;
        };

        ///
        /// \brief streaming operators.
        ///
        NANO_PUBLIC std::ostream& operator<<(std::ostream&, const table_t&);

        ///
        /// \brief comparison operators.
        ///
        NANO_PUBLIC bool operator==(const row_t&, const row_t&);
        NANO_PUBLIC bool operator==(const cell_t&, const cell_t&);
        NANO_PUBLIC bool operator==(const table_t&, const table_t&);

        ///
        /// \brief collects & formats tabular data for ASCII display.
        ///
        struct NANO_PUBLIC table_t
        {
                enum class sorting
                {
                        asc,
                        desc
                };

                table_t() = default;

                ///
                /// \brief remove all rows, but keeps the header
                ///
                void clear();

                ///
                /// \brief append a row as a header, as a data or as a delimeter row
                ///
                row_t& delim();
                row_t& header();
                row_t& append();

                ///
                /// \brief print table
                ///
                std::ostream& print(std::ostream&) const;

                ///
                /// \brief check if equal with another table
                ///
                bool equals(const table_t&) const;

                ///
                /// \brief (stable) sort the table using the given row-based comparison operator
                ///
                template <typename toperator>
                void sort(const toperator&);

                ///
                /// \brief (stable) sort the table using the given columns
                ///
                template <typename toperator>
                void sort(const toperator&, const indices_t& columns);

                ///
                /// \brief (stable) sort the table using the given columns
                ///
                template <typename tvalue>
                void sort(const sorting, const indices_t& columns);

                ///
                /// \brief mark row-wise the selected columns with the given operator
                ///
                template <typename tmarker>
                void mark(const tmarker& marker, const char* marker_string = " (*)");

                ///
                /// \brief save/load to/from CSV files using the given separator
                /// the header is always written/read
                ///
                bool save(const string_t& path, const string_t& delim = ";") const;
                bool load(const string_t& path, const string_t& delim = ";", const bool load_header = true);

                ///
                /// \brief access functions
                ///
                size_t cols() const;
                size_t rows() const;
                const row_t& row(const size_t r) const { return m_rows.at(r); }

        private:

                // attributes
                std::vector<row_t>      m_rows;
        };

        template <typename toperator>
        void table_t::sort(const toperator& comp)
        {
                std::stable_sort(m_rows.begin(), m_rows.end(), comp);
        }

        template <typename toperator>
        void table_t::sort(const toperator& comp, const indices_t& columns)
        {
                sort([&] (const auto& row1, const auto& row2)
                {
                        for (const auto col : columns)
                        {
                                if (comp(row1.value(col), row2.value(col)))
                                {
                                        return true;
                                }
                                else if (comp(row2.value(col), row1.value(col)))
                                {
                                        return false;
                                }
                        }
                        return true;
                });
        }

        template <typename tvalue>
        void table_t::sort(const sorting type, const indices_t& columns)
        {
                switch (type)
                {
                case sorting::asc:
                        sort(nano::make_less_from_string<tvalue>(), columns);
                        break;

                case sorting::desc:
                        sort(nano::make_greater_from_string<tvalue>(), columns);
                        break;
                }
        }

        template <typename tmarker>
        void table_t::mark(const tmarker& marker, const char* marker_string)
        {
                for (auto& row : m_rows)
                {
                        const auto sel_cols = marker(row);
                        for (const auto& col : sel_cols)
                        {
                                row.mark(col, marker_string);
                        }
                }
        }
}
