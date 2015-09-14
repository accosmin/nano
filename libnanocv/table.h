#pragma once

#include "table_row.h"
#include "table_header.h"
#include "libnanocv/arch.h"
#include <functional>

namespace ncv
{
        ///
        /// \brief collects & formats for ASCII display tabular data
        ///
        class NANOCV_PUBLIC table_t
        {
        public:

                typedef std::function
                <
                        bool(const string_t&,
                             const string_t&)
                >                                       comparator_t;

                ///
                /// \brief return the selected column index from all columns of a row
                ///
                typedef std::function
                <
                        size_t(const strings_t&)
                >                                       marker_t;

                enum class sorting
                {
                        ascending,
                        descending
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
                table_row_t& append(const string_t& name);

                ///
                /// \brief sort the table by the given column and using the given comparison operator
                ///
                bool sort(size_t col, const comparator_t& comp);

                ///
                /// \brief sort by transforming to numeric values the given column
                ///
                bool sort_as_number(size_t col, sorting mode);

                ///
                /// \brief mark row-wise the selected element with the given operator
                ///
                bool mark(const marker_t& op, const char* marker_string = " (*)");

                ///
                /// \brief mark row-wise the minimum element
                ///
                bool mark_min_number(const char* marker_string = " (*)");

                ///
                /// \brief pretty-print its content
                ///
                bool print(std::ostream& os,
                           const char table_delim = '=',
                           const char row_delim = '.', bool use_row_delim = false) const;

        private:

                ///
                /// \brief retrieve the number of columns
                ///
                std::size_t cols() const { return m_header.size(); }

                ///
                /// \brief retrieve the (current) number of rows
                ///
                std::size_t rows() const { return m_rows.size(); }

                ///
                /// \brief border size
                ///
                std::size_t border() const;

                ///
                /// \brief compute the size of each value column
                ///
                std::vector<std::size_t> value_colsizes() const;

                ///
                /// \brief compute the size of the name column
                ///
                size_t name_colsize() const;

        private:

                // attributes
                string_t                        m_title;        ///<
                table_header_t                  m_header;       ///<
                std::vector<table_row_t>        m_rows;         ///<
        };
}

