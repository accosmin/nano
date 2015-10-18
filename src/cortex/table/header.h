#pragma once

#include "string.h"
#include "text/to_string.hpp"

namespace cortex
{
        ///
        /// \brief a header in the table
        ///
        class table_header_t
        {
        public:

                ///
                /// \brief append a column value
                ///
                template
                <
                        typename tvalue
                >
                table_header_t& operator<<(tvalue value)
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
}

