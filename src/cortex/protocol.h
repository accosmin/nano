#pragma once

#include "text/enum_string.hpp"

namespace zob
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
namespace zob
{
        template <>
        inline std::map<zob::protocol, std::string> enum_string<zob::protocol>()
        {
                return
                {
                        { zob::protocol::train,      "train" },
                        { zob::protocol::test,       "test" }
                };
        }
}


