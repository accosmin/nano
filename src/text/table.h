#pragma once

#include "scalar.h"
#include "table_row.h"
#include "from_string.h"
#include "table_header.h"
#include <algorithm>

namespace nano
{
        ///
        /// \brief collects & formats for ASCII display tabular data
        ///
        class NANO_PUBLIC table_t
        {
        public:

                enum class sorting
                {
                        asc,
                        desc
                };

                ///
                /// \brief constructor
                ///
                explicit table_t(const string_t& title);

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
                template <typename tname>
                table_row_t& append(const tname& name);

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
                /// \brief pretty-print its content
                ///
                void print(std::ostream& os, const bool use_row_delim = false) const;

                ///
                /// \brief access functions
                ///
                std::size_t cols() const;
                std::size_t rows() const;
                const table_header_t& header() const;
                const table_row_t& row(const std::size_t index) const;

        private:

                ///
                /// \brief compute the size of each value column
                ///
                sizes_t value_colsizes() const;

                ///
                /// \brief compute the size of the name column
                ///
                size_t name_colsize() const;

                ///
                /// \brief print a row delimiter
                ///
                void print_row_delim(std::ostream& os) const;

        private:

                // attributes
                string_t                        m_title;        ///<
                table_header_t                  m_header;       ///<
                std::vector<table_row_t>        m_rows;         ///<
        };

        template <typename tname>
        table_row_t& table_t::append(const tname& name)
        {
                m_rows.emplace_back(to_string(name));
                return *m_rows.rbegin();
        }

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

