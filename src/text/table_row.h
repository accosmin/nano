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

                enum class storage
                {
                        data,           ///< populated with data
                        delim,          ///< used as delimiter
                };

                ///
                /// \brief constructor
                ///
                table_row_t(const storage t = storage::data) :
                        m_type(t)
                {
                }

                ///
                /// \brief append a column value
                ///
                template <typename tvalue>
                table_row_t& operator<<(tvalue value)
                {
                        m_values.emplace_back(to_string(value));
                        m_markings.emplace_back();
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

                ///
                /// \brief storage type
                ///
                auto type() const { return m_type; }

        private:

                // attributes
                storage         m_type;
                strings_t       m_values;       ///< column values
                strings_t       m_markings;     ///< column marking (e.g. min|max decoration)
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
