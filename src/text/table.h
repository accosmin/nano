#pragma once

#include "arch.h"
#include "scalar.h"
#include <algorithm>
#include "table_row.h"
#include "from_string.h"
#include "table_header.h"

namespace nano
{
        struct table_t;

        ///
        /// \brief streaming operator.
        ///
        NANO_PUBLIC std::ostream& operator<<(std::ostream&, const table_t&);

        ///
        /// \brief comparison operator.
        ///
        NANO_PUBLIC bool operator==(const table_t& t1, const table_t& t2);

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

                ///
                /// \brief constructor
                ///
                explicit table_t();

                ///
                /// \brief remove all rows, but keeps the header
                ///
                void clear();

                ///
                /// \brief access header
                ///
                table_header_t& header();

                ///
                /// \brief append a new row
                ///
                table_row_t& append(const table_row_t::storage type = table_row_t::storage::data);

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
                const table_header_t& header() const;
                const table_row_t& row(const std::size_t index) const;

        private:

                // attributes
                table_header_t                  m_header;       ///<
                std::vector<table_row_t>        m_rows;         ///<
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
