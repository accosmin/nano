#pragma once

#include "to_string.h"

namespace nano
{
        ///
        /// \brief a header in the table.
        ///
        class table_header_t
        {
        public:

                ///
                /// \brief append a column value
                ///
                template <typename tvalue>
                table_header_t& operator<<(const tvalue value)
                {
                        m_values.emplace_back(to_string(value));
                        return *this;
                }

                ///
                /// \brief retrieve the column values
                ///
                const auto& values() const { return m_values; }
                const auto& value(const size_t i) const { return m_values.at(i); }
                auto& value(const size_t i) { return m_values.at(i); }

                ///
                /// \brief retrieve the number of columns
                ///
                size_t size() const { return m_values.size(); }

        private:

                // attributes
                strings_t       m_values;       ///< column values
        };

        ///
        /// \brief comparison operator.
        ///
        inline bool operator==(const table_header_t& h1, const table_header_t& h2)
        {
                return h1.values() == h2.values();
        }
}
