#pragma once

#include "util/text.h"
#include <stdexcept>

namespace ncv
{
        ///
        /// \brief regularization methods
        ///
        enum class regularizer
        {
                none = 0,               ///< no regularization term
                l2norm,                 ///< L2-norm regularization
                variational             ///< penalize high loss variation
        };

        // string cast for enumerations
        namespace text
        {
                template <>
                inline std::string to_string(regularizer type)
                {
                        switch (type)
                        {
                        case regularizer::none:         return "none";
                        case regularizer::l2norm:       return "l2";
                        case regularizer::variational:  return "var";
                        default:                        return "none";
                        }
                }

                template <>
                inline regularizer from_string<regularizer>(const std::string& string)
                {
                        if (string == "none")           return regularizer::none;
                        if (string == "l2")             return regularizer::l2norm;
                        if (string == "var")            return regularizer::variational;
                        throw std::invalid_argument("invalid regularizer <" + string + ">!");
                        return regularizer::none;
                }
        }
}


