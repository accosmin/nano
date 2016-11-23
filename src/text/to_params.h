#pragma once

#include "to_string.h"

namespace nano
{
        ///
        /// \brief encode parameter by name: [name1=value1[,name2=value2[...]]
        ///
        template <typename tvalue>
        string_t to_params(const char* name, const tvalue& value)
        {
                return string_t(name) + "=" + to_string(value);
        }

        template <typename tvalue, typename... tvalues>
        string_t to_params(const char* name, const tvalue& value, const tvalues&... values)
        {
                return to_params(name, value) + "," + to_params(values...);
        }

        template <typename tvalue, typename... tvalues>
        string_t to_params(const string_t& params, const char* name, const tvalue& value, const tvalues&... values)
        {
                return params + (params.empty() ? "" : ",") + to_params(name, value, values...);
        }
}

