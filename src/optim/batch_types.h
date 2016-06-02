#pragma once

#include "text/enum_string.hpp"

namespace nano
{
        ///
        /// \brief batch optimization methods
        ///
        enum class batch_optimizer
        {
                GD,                             ///< gradient descent
                CGD,                            ///< conjugate gradient descent (default version)
                LBFGS,                          ///< limited-memory BFGS

                CGD_HS,                         ///< various conjugate gradient descent versions
                CGD_FR,
                CGD_PRP,
                CGD_CD,
                CGD_LS,
                CGD_DY,
                CGD_N,
                CGD_DYCD,
                CGD_DYHS
        };

        template <>
        inline std::map<batch_optimizer, std::string> enum_string<batch_optimizer>()
        {
                return
                {
                        { batch_optimizer::GD,           "gd" },
                        { batch_optimizer::CGD,          "cgd" },
                        { batch_optimizer::LBFGS,        "lbfgs" },
                        { batch_optimizer::CGD_HS,       "cgd-hs" },
                        { batch_optimizer::CGD_FR,       "cgd-fr" },
                        { batch_optimizer::CGD_PRP,      "cgd-prp" },
                        { batch_optimizer::CGD_CD,       "cgd-cd" },
                        { batch_optimizer::CGD_LS,       "cgd-ls" },
                        { batch_optimizer::CGD_DY,       "cgd-dy" },
                        { batch_optimizer::CGD_N,        "cgd-n" },
                        { batch_optimizer::CGD_DYCD,     "cgd-dycd" },
                        { batch_optimizer::CGD_DYHS,     "cgd-dyhs" }
                };
        }
}
