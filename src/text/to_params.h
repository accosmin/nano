#pragma once

#include "to_string.h"

namespace nano
{
        ///
        /// \brief encode parameter by name: [name1=value1[,name2=value2[...]]
        ///
        template
        <
                class tvalue
        >
        string_t to_params(const char* name, const tvalue& value)
        {
                return string_t(name) + "=" + to_string(value);
        }

        template
        <
                class tvalue,
                class... tvalues
        >
        string_t to_params(const char* name, const tvalue& value, const tvalues&... values)
        {
                return to_params(name, value) + "," + to_params(values...);
        }

        inline string_t concat_params(const string_t& params1, const string_t& params2)
        {
                return params1.empty() ? params2 : (params1 + "," + params2);
        }
}

