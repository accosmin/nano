#pragma once

#include "stringi.h"

namespace nano
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
        inline string_t align(const string_t& str, const std::size_t str_size,
                const alignment mode = alignment::left, const char fill_char = ' ')
        {
                const auto fill_size = (str.size() > str_size) ? (0) : (str_size - str.size());

                switch (mode)
                {
                case alignment::center:
                        return string_t(fill_size / 2, fill_char) +
                               str +
                               string_t(fill_size - fill_size / 2, fill_char);

                case alignment::right:
                        return string_t(fill_size, fill_char) +
                               str;

                case alignment::left:
                default:
                        return str +
                               string_t(fill_size, fill_char);
                }
        }
}

