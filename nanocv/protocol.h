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
                inline std::string to_string(protocol type)
                {
                        switch (type)
                        {
                        case protocol::train:           return "train";
                        case protocol::test:            return "test";
                        default:                        return "train";
                        }
                }

                template <>
                inline protocol from_string<protocol>(const std::string& string)
                {
                        if (string == "train")          return protocol::train;
                        if (string == "test")           return protocol::test;
                        throw std::invalid_argument("invalid protocol <" + string + ">!");
                        return protocol::train;
                }
        }
}


