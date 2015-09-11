#pragma once

#include "text.hpp"
#include "libnanocv/arch.h"
#include <functional>

namespace ncv
{
        ///
        /// \brief collects & formats for ASCII display tabular data
        ///
        class NANOCV_PUBLIC tabulator_t
        {
        public:

                typedef std::string                     string_t;
                typedef std::vector<string_t>           strings_t;

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
                /// \brief a header in the table
                ///
                class header_t
                {
                public:

                        ///
                        /// \brief append a column value
                        ///
                        template
                        <
                                typename tvalue
                        >
                        header_t& operator<<(tvalue value)
                        {
                                m_values.emplace_back(text::to_string(value));
                                return *this;
                        }

                        ///
                        /// \brief retrieve the column values
                        ///
                        const strings_t& values() const { return m_values; }
                        const string_t& operator[](size_t i) const { return m_values[i]; }

                        ///
                        /// \brief retrieve the number of columns
                        ///
                        size_t size() const { return m_values.size(); }

                private:

                        // attributes
                        strings_t       m_values;       ///< column values
                };

                ///
                /// \brief a row in the table
                ///
                class row_t
                {
                public:

                        ///
                        /// \brief constructor
                        ///
                        explicit row_t(const string_t& name)
                                :       m_name(name)
                        {
                        }

                        ///
                        /// \brief append a column value
                        ///
                        template
                        <
                                typename tvalue
                        >
                        row_t& operator<<(tvalue value)
                        {
                                m_values.emplace_back(text::to_string(value));
                                return *this;
                        }

                        ///
                        /// \brief retrieve the row name
                        ///
                        const string_t& name() const { return m_name; }

                        ///
                        /// \brief retrieve the column values
                        ///
                        const strings_t& values() const { return m_values; }
                        const string_t& operator[](size_t i) const { return m_values[i]; }

                        string_t& operator[](size_t i) { return m_values[i]; }

                        ///
                        /// \brief retrieve the number of columns
                        ///
                        size_t size() const { return m_values.size(); }

                private:

                        // attributes
                        string_t        m_name;         ///< row name
                        strings_t       m_values;       ///< column values
                };

                ///
                /// \brief constructor
                ///
                explicit tabulator_t(const string_t& title);

                ///
                /// \brief remove all rows, but keeps the header
                ///
                void clear();

                ///
                /// \brief access header
                ///
                header_t& header();

                ///
                /// \brief append a new row
                ///
                row_t& append(const string_t& name);

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
                string_t                m_title;        ///<
                header_t                m_header;       ///<
                std::vector<row_t>      m_rows;         ///<
        };
}

