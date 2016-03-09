#pragma once

#include "to_string.hpp"
#include <vector>

namespace zob
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
                explicit table_row_t(const std::string& name)
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
                        m_values.emplace_back(zob::to_string(value));
                        return *this;
                }

                ///
                /// \brief retrieve the row name
                ///
                const auto& name() const { return m_name; }

                ///
                /// \brief retrieve the column values
                ///
                const auto& operator[](size_t i) const { return m_values[i]; }
                auto& operator[](size_t i) { return m_values[i]; }

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
                std::string                     m_name;         ///< row name
                std::vector<std::string>        m_values;       ///< column values
        };
}

