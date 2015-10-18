#pragma once

#include "text/enum_string.hpp"

namespace cortex
{
        ///
        /// \brief machine learning protocols
        ///
        enum class protocol
        {
                train = 0,              ///< training
                test                    ///< testing
        };
}

// string cast for enumerations
namespace text
{
        template <>
        inline std::map<cortex::protocol, std::string> enum_string<cortex::protocol>()
        {
                return
                {
                        { cortex::protocol::train,      "train" },
                        { cortex::protocol::test,       "test" }
                };
        }
}


