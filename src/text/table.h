#pragma once

#include "arch.h"
#include "table_row.h"
#include "table_header.h"
#include <cassert>
#include <algorithm>

namespace text
{
        ///
        /// \brief collects & formats for ASCII display tabular data
        ///
        class NANOCV_PUBLIC table_t
        {
        public:

                ///
                /// \brief constructor
                ///
                explicit table_t(const std::string& title);

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
                table_row_t& append(const std::string& name);

                ///
                /// \brief sort the table using the given row comparison
                ///
                template
                <
                        typename trow_comp
                >
                void sort(const trow_comp& comp)
                {
                        std::stable_sort(m_rows.begin(), m_rows.end(), comp);
                }

                ///
                /// \brief mark row-wise the selected element with the given operator
                ///
                template
                <
                        typename trow_mark
                >
                void mark(const trow_mark& mark, const char* marker_string = " (*)")
                {
                        for (auto& row : m_rows)
                        {
                                const auto col = mark(row);
                                assert(col < cols());
                                row[col] += marker_string;
                        }
                }

                ///
                /// \brief pretty-print its content
                ///
                void print(std::ostream& os, const bool use_row_delim = false) const;

                ///
                /// \brief retrieve the number of columns
                ///
                std::size_t cols() const;

                ///
                /// \brief retrieve the (current) number of rows
                ///
                std::size_t rows() const;

        private:

                ///
                /// \brief compute the size of each value column
                ///
                std::vector<std::size_t> value_colsizes() const;

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
                std::string                     m_title;        ///<
                table_header_t                  m_header;       ///<
                std::vector<table_row_t>        m_rows;         ///<
        };
}

