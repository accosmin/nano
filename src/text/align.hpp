#pragma once

#include <string>

namespace text
{
        ///
        /// \brief text alignment options
        ///
        enum class alignment : int
        {
                left,
                center,
                right
        };

        ///
        /// \brief align a string to fill the given size (if possible)
        ///
        inline std::string align(const std::string& str, const std::size_t str_size,
                const alignment mode = alignment::left, const char fill_char = ' ')
        {
                const auto fill_size = (str.size() > str_size) ? (0) : (str_size - str.size());

                switch (mode)
                {
                case alignment::center:
                        return std::string(fill_size / 2, fill_char) +
                               str +
                               std::string(fill_size - fill_size / 2, fill_char);

                case alignment::right:
                        return std::string(fill_size, fill_char) +
                               str;

                case alignment::left:
                default:
                        return str +
                               std::string(fill_size, fill_char);
                }
        }
}

