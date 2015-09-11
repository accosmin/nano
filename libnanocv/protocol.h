#pragma once

#include "libnanocv/text/enum_string.hpp"

namespace ncv
{
        ///
        /// \brief machine learning protocols
        ///
        enum class protocol
        {
                train = 0,              ///< training
                test                    ///< testing
        };

        // string cast for enumerations
        namespace text
        {
                template <>
                inline std::map<protocol, std::string> enum_string<protocol>()
                {
                        return
                        {
                                { protocol::train,      "train" },
                                { protocol::test,       "test" }
                        };
                }
        }
}


