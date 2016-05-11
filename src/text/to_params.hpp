#pragma once

#include "to_string.hpp"

namespace nano
{
        ///
        /// \brief decode parameter by name: [name1=value1[,name2=value2[...]]
        /// the default value is returned if the parameter cannot be found or is invalid.
        ///
        template
        <
                class tvalue
        >
        std::string to_params(const char* name, const tvalue& value)
        {
                return std::string(name) + "=" + to_string(value);
        }

        template
        <
                class tvalue,
                class... tvalues
        >
        std::string to_params(const char* name, const tvalue& value, const tvalues&... values)
        {
                return to_params(name, value) + "," + to_params(values...);
        }
}

