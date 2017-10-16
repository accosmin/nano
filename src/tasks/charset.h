#pragma once

#include "text/enum_string.h"

namespace nano
{
        enum class charset_type
        {
                digit,          ///< 0-9
                lalpha,         ///< a-z
                ualpha,         ///< A-Z
                alpha,          ///< a-zA-Z
                alphanum,       ///< A-Za-z0-9
        };

        template <>
        inline enum_map_t<charset_type> enum_string<charset_type>()
        {
                return
                {
                        { charset_type::digit,          "digit" },
                        { charset_type::lalpha,         "lalpha" },
                        { charset_type::ualpha,         "ualpha" },
                        { charset_type::alpha,          "alpha" },
                        { charset_type::alphanum,       "alphanum" }
                };
        }
}
