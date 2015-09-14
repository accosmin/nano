#pragma once

#include <string>

namespace ncv
{
        ///
        /// \brief text alignment options
        ///
        enum class align : int
        {
                left,
                center,
                right
        };

        namespace text
        {
                ///
                /// \brief align a string to fill the given size (if possible)
                ///
                inline std::string align(const std::string& str, const std::size_t str_size,
                        const ncv::align alignment = align::left, const char fill_char = ' ')
                {
                        const std::size_t fill_size = str.size() > str_size ? 0 : str_size - str.size();

                        switch (alignment)
                        {
                        case ncv::align::center:
                                return std::string(fill_size / 2, fill_char) +
                                       str +
                                       std::string(fill_size - fill_size / 2, fill_char);

                        case ncv::align::right:
                                return std::string(fill_size, fill_char) +
                                       str;

                        case ncv::align::left:
                        default:
                                return str +
                                       std::string(fill_size, fill_char);
                        }
                }
        }
}

