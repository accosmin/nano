#pragma once

#include "text/enum_string.hpp"

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
}

// string cast for enumerations
namespace text
{
        template <>
        inline std::map<ncv::protocol, std::string> enum_string<ncv::protocol>()
        {
                return
                {
                        { ncv::protocol::train,      "train" },
                        { ncv::protocol::test,       "test" }
                };
        }
}


