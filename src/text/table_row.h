#pragma once

#include "to_string.h"

namespace nano
{
        ///
        /// \brief a row in the table.
        ///
        class table_row_t
        {
        public:

                ///
                /// \brief append a column value
                ///
                template <typename tvalue>
                table_row_t& operator<<(tvalue value)
                {
                        m_values.emplace_back(nano::to_string(value));
                        m_markings.push_back(string_t());
                        return *this;
                }

                ///
                /// \brief retrieve the column values & markings
                ///
                const auto& values() const { return m_values; }
                const auto& value(const size_t i) const { return m_values.at(i); }
                auto& value(const size_t i) { return m_values.at(i); }

                auto begin() const { return m_values.cbegin(); }
                auto end() const { return m_values.cend(); }

                const auto& marking(const size_t i) const { return m_markings.at(i); }
                auto& marking(const size_t i) { return m_markings.at(i); }

                ///
                /// \brief retrieve the number of columns
                ///
                size_t size() const { return m_values.size(); }

        private:

                // attributes
                strings_t               m_values;       ///< column values
                strings_t               m_markings;     ///< column marking (e.g. min|max decoration)
        };

        ///
        /// \brief comparison operator
        ///
        inline bool operator==(const table_row_t& r1, const table_row_t& r2)
        {
                return r1.values() == r2.values();
        }
        inline bool operator!=(const table_row_t& r1, const table_row_t& r2)
        {
                return !(r1 == r2);
        }
}
