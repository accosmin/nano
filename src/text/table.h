#pragma once

#include "arch.h"
#include "scalar.h"
#include <algorithm>
#include "algorithm.h"
#include "from_string.h"

namespace nano
{
        struct table_t;

        struct cell_t
        {
                cell_t();

                template <typename tvalue>
                cell_t(const tvalue value, const size_t span = 1, const alignment align = alignment::left) :
                        m_data(to_string(value)),
                        m_span(span),
                        m_align(align)
                {
                }

                bool empty() const { return m_data.empty(); }
                void print(std::ostream&, const size_t maximum) const;

                // attributes
                string_t                m_data;
                string_t                m_mark;
                size_t                  m_span;
                alignment               m_align;
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

                row_t& operator<<(const cell_t& cell);

                size_t cols() const;
                cell_t find(const size_t col) const;

                // attributes
                type                    m_type;
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
                                row.marking(col) = marker_string;
                        }
                }
        }
}
