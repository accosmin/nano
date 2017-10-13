#pragma once

#include "arch.h"
#include "scalar.h"
#include <cassert>
#include <algorithm>
#include "to_string.h"
#include "from_string.h"

namespace nano
{
        ///
        /// \brief cell in a table.
        ///
        struct NANO_PUBLIC cell_t
        {
                cell_t();
                cell_t(const string_t& data, const size_t span, const alignment);

                const auto& data() const { return m_data; }
                bool empty() const { return data().empty(); }
                void print(std::ostream&, const size_t maximum) const;

                // attributes
                string_t                m_data;
                string_t                m_mark;
                size_t                  m_span;
                alignment               m_alignment;
        };

        ///
        /// \brief row in a table.
        ///
        struct NANO_PUBLIC row_t
        {
                enum class mode
                {
                        data,           ///<
                        delim,          ///< delimiting row
                        header,         ///< header (not considered for operations like sorting or marking)
                };

                row_t(const mode t = mode::data);

                template <typename tscalar>
                row_t& operator<<(const tscalar value)
                {
                        m_cells.emplace_back(to_string(value), colspan(), align());
                        return *this;
                }

                ///
                /// \brief return the number of columns taking into account column spanning
                ///
                size_t cols() const;

                ///
                /// \brief mark a column (finds the right cell taking into account column spanning)
                ///
                void mark(const size_t col, const string_t& marker);

                ///
                /// \brief find the a cell taking into account column spanning
                ///
                cell_t* find(const size_t col);
                const cell_t* find(const size_t col) const;

                ///
                /// \brief access functions
                ///
                auto type() const { return m_type; }
                const auto& cells() const { return m_cells; }
                const auto& cell(const size_t c) const { return m_cells.at(c); }

                ///
                /// \brief select the columns that satisfy the given operator
                ///
                template <typename tscalar, typename toperator>
                auto select_cols(const row_t& row, const toperator& op)
                {
                        indices_t indices;
                        if (row.type() == row_t::mode::data)
                        {
                                for (size_t col = 0, cols = row.cols(); col < cols; ++ col)
                                {
                                        try
                                        {
                                                const cell_t* cell = row.find(col);
                                                assert(cell);
                                                if (op(nano::from_string<tscalar>(cell->data())))
                                                {
                                                        indices.push_back(col);
                                                }
                                        }
                                        catch (std::exception&) {}
                                }
                        }
                        return indices;
                }

                size_t colspan() const { return m_colspan; }
                alignment align() const { return m_alignment; }
                row_t& colspan(const size_t span) { m_colspan = span; return *this; }
                row_t& align(const alignment align) { m_alignment = align; return *this; }

        private:

                // attributes
                mode                    m_type;
                size_t                  m_colspan;      ///< current column span
                alignment               m_alignment;    ///< current cell alignment
                std::vector<cell_t>     m_cells;
        };

        struct table_t;

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

        enum class sorting
        {
                asc,
                desc
        };

        ///
        /// \brief collects & formats tabular data for ASCII display.
        ///
        struct NANO_PUBLIC table_t
        {
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
                template <typename tscalar>
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
                sort([&] (const row_t& row1, const row_t& row2)
                {
                        if (row1.type() == row_t::mode::data && row2.type() == row_t::mode::data)
                        {
                                assert(row1.cols() == row2.cols());
                                for (const auto col : columns)
                                {
                                        assert(row1.find(col) && row2.find(col));
                                        const auto* cell1 = row1.find(col);
                                        const auto* cell2 = row2.find(col);
                                        if (comp(cell1->data(), cell2->data()))
                                        {
                                                return true;
                                        }
                                        else if (comp(cell1->data(), cell2->data()))
                                        {
                                                return false;
                                        }
                                }
                        }
                        return true;
                });
        }

        template <typename tscalar>
        void table_t::sort(const sorting type, const indices_t& columns)
        {
                switch (type)
                {
                case sorting::asc:
                        sort(nano::make_less_from_string<tscalar>(), columns);
                        break;

                case sorting::desc:
                        sort(nano::make_greater_from_string<tscalar>(), columns);
                        break;
                }
        }

        template <typename tmarker>
        void table_t::mark(const tmarker& marker, const char* marker_string)
        {
                for (auto& row : m_rows)
                {
                        for (const auto col : marker(row))
                        {
                                row.mark(col, marker_string);
                        }
                }
        }

        namespace detail
        {
                template <typename tscalar>
                auto min_element(const row_t& row)
                {
                        const auto op = nano::make_less_from_string<tscalar>();
                        const auto it = std::min_element(row.begin(), row.end(), op);
                        assert(it != row.end());
                        return it;
                }

                template <typename tscalar>
                auto max_element(const row_t& row)
                {
                        const auto op = nano::make_less_from_string<tscalar>();
                        const auto it = std::max_element(row.begin(), row.end(), op);
                        assert(it != row.end());
                        return it;
                }
        }

        ///
        /// \brief select the column with the minimum value
        ///
        template <typename tscalar>
        auto make_table_mark_minimum_col()
        {
                return [=] (const row_t& row) -> indices_t
                {
                        const auto it = detail::min_element<tscalar>(row);
                        return { static_cast<size_t>(it - row.begin()) };
                };
        }

        ///
        /// \brief select the column with the maximum value
        ///
        template <typename tscalar>
        auto make_table_mark_maximum_col()
        {
                return [=] (const row_t& row) -> indices_t
                {
                        const auto it = detail::max_element<tscalar>(row);
                        return { static_cast<size_t>(it - row.begin()) };
                };
        }

        ///
        /// \brief select the columns within [0, epsilon] from the maximum value
        ///
        template <typename tscalar>
        auto make_table_mark_maximum_epsilon_cols(const tscalar epsilon)
        {
                return [=] (const row_t& row)
                {
                        const auto it = detail::max_element<tscalar>(row);
                        const auto max = nano::from_string<tscalar>(*it);
                        const auto thres = max - epsilon;

                        return detail::select_cols<tscalar>(row, [thres] (const auto& val) { return val >= thres; });
                };
        }

        ///
        /// \brief select the columns within [0, epsilon] from the minimum value
        ///
        template <typename tscalar>
        auto make_table_mark_minimum_epsilon_cols(const tscalar epsilon)
        {
                return [=] (const row_t& row)
                {
                        const auto it = detail::min_element<tscalar>(row);
                        const auto min = nano::from_string<tscalar>(*it);
                        const auto thres = min + epsilon;

                        return detail::select_cols<tscalar>(row, [thres] (const auto& val) { return val <= thres; });
                };
        }

        ///
        /// \brief select the columns within [0, percentage]% from the maximum value
        ///
        template <typename tscalar>
        auto make_table_mark_maximum_percentage_cols(const tscalar percentage)
        {
                return [=] (const row_t& row)
                {
                        assert(percentage >= tscalar(1));
                        assert(percentage <= tscalar(99));

                        const auto it = detail::max_element<tscalar>(row);
                        const auto max = nano::from_string<tscalar>(*it);
                        const auto thres = max - percentage * (max < 0 ? -max : +max) / tscalar(100);

                        return detail::select_cols<tscalar>(row, [thres] (const auto& val) { return val >= thres; });
                };
        }

        ///
        /// \brief select the columns within [0, percentage]% from the minimum value
        ///
        template <typename tscalar>
        auto make_table_mark_minimum_percentage_cols(const tscalar percentage)
        {
                return [=] (const row_t& row)
                {
                        assert(percentage >= tscalar(1));
                        assert(percentage <= tscalar(99));

                        const auto it = detail::min_element<tscalar>(row);
                        const auto min = nano::from_string<tscalar>(*it);
                        const auto thres = min + percentage * (min < 0 ? -min : +min) / tscalar(100);

                        return detail::select_cols<tscalar>(row, [thres] (const auto& val) { return val <= thres; });
                };
        }
}
