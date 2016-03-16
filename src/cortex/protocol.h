#pragma once

#include "text/enum_string.hpp"

namespace nano
{
        ///
        /// \brief dataset splitting protocol
        ///
        enum class protocol
        {
                train = 0,              ///< training
                valid,                  ///< validation (for tuning hyper-parameters)
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
                        { nano::protocol::train,        "train" },
                        { nano::protocol::valid,        "valid" },
                        { nano::protocol::test,         "test" }
                };
        }
}


