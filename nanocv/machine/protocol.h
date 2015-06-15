#pragma once

#include "text.h"
#include <stdexcept>

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


