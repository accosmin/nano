#pragma once

#include "to_string.hpp"

namespace text
{
        ///
        /// \brief compact a list of values into a string using the given "glue" string
        ///
        template
        <
                typename tcontainer,
                typename tvalue = decltype(*std::begin(tcontainer()))
        >
        std::string concatenate(const tcontainer& values, const std::string& glue = ",")
        {
                std::string ret;
                for (auto value : values)
                {
                        ret += to_string(value) + glue;
                };

                return ret.empty() ? ret : ret.substr(0, ret.size() - glue.size());
        }
}

