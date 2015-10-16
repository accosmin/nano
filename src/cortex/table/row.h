#pragma once

#include "cortex/string.h"
#include "text/to_string.hpp"

namespace ncv
{
        ///
        /// \brief a row in the table.
        ///
        class table_row_t
        {
        public:

                ///
                /// \brief constructor
                ///
                explicit table_row_t(const string_t& name)
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
                table_row_t& operator<<(tvalue value)
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
                const string_t& operator[](size_t i) const { return m_values[i]; }
                string_t& operator[](size_t i) { return m_values[i]; }

                ///
                /// \brief retrieve the column value range
                ///
                auto begin() const { return m_values.cbegin(); }
                auto end() const { return m_values.cend(); }

                ///
                /// \brief retrieve the number of columns
                ///
                size_t size() const { return m_values.size(); }

        private:

                // attributes
                string_t        m_name;         ///< row name
                strings_t       m_values;       ///< column values
        };
}

