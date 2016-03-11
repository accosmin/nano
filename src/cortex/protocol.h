#pragma once

#include "text/enum_string.hpp"

namespace nano
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
namespace nano
{
        template <>
        inline std::map<nano::protocol, std::string> enum_string<nano::protocol>()
        {
                return
                {
                        { nano::protocol::train,      "train" },
                        { nano::protocol::test,       "test" }
                };
        }
}


