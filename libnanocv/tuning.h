#pragma once

#include "text.h"
#include <stdexcept>

namespace ncv
{
        ///
        /// \brief methods to tune the regularization weights
        ///
        enum class reg_tuning
        {
                none,
                log10_search,           ///< branch and bound search on the log10 scale
                continuation            ///< continuation methods (smooth to
        };

        // string cast for enumerations
        namespace text
        {
                template <>
                inline std::string to_string(reg_tuning type)
                {
                        switch (type)
                        {
                        case reg_tuning::log10_search:  return "tune";
                        case reg_tuning::continuation:  return "cont";
                        default:                        return "none";
                        }
                }

                template <>
                inline reg_tuning from_string<reg_tuning>(const std::string& string)
                {
                        if (string == "none")           return reg_tuning::none;
                        if (string == "tune")           return reg_tuning::log10_search;
                        if (string == "cont")           return reg_tuning::continuation;
                        throw std::invalid_argument("invalid regularization tuning <" + string + ">!");
                        return reg_tuning::none;
                }
        }
}


