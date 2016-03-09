#pragma once

#include "to_string.hpp"
#include <vector>

namespace zob
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
                        m_values.emplace_back(zob::to_string(value));
                        return *this;
                }

                ///
                /// \brief retrieve the column values
                ///
                const auto& values() const { return m_values; }
                const auto& operator[](size_t i) const { return m_values[i]; }

                ///
                /// \brief retrieve the number of columns
                ///
                size_t size() const { return m_values.size(); }

        private:

                // attributes
                std::vector<std::string>        m_values;       ///< column values
        };
}

