#pragma once

#include "to_string.hpp"

namespace nano
{
        ///
        /// \brief compact a list of values into a string using the given "glue" string
        ///
        template
        <
                typename tcontainer,
                typename tvalue = decltype(*std::begin(tcontainer()))
        >
        string_t concatenate(const tcontainer& values, const string_t& glue = ",")
        {
                string_t ret;
                for (auto value : values)
                {
                        ret += to_string(value) + glue;
                };

                return ret.empty() ? ret : ret.substr(0, ret.size() - glue.size());
        }
}

