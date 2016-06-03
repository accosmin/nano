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

        ///
        /// \brief initial step length strategy
        ///     see "Numerical optimization", Nocedal & Wright, 2nd edition, p.59
        ///
        enum class ls_initializer
        {
                unit,                           ///< 1.0 (useful for quasi-Newton and Newton methods)
                consistent,                     ///< consistent first-order change in the function
                quadratic                       ///< quadratic local interpolation (previous & current position)
        };

        ///
        /// \brief line-search strategy
        ///
        enum class ls_strategy
        {
                backtrack_armijo,               ///< backtracking with sufficient decrease (Armijo)
                backtrack_wolfe,                ///< + backtracking with suficient curvature (Wolfe)
                backtrack_strong_wolfe,         ///< + backtracking with sufficient curvature (strong Wolfe)

                // see "Numerical optimization", Nocedal & Wright, 2nd edition, p.60-61 - strong Wolfe only
                interpolation,                  ///< bisection/quadratic/cubic for zooming

                // see CG_DESCENT, Hager & Zhang, 2005 - regular and approximate Wolfe only
                cg_descent                      ///< CG_DESCENT
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

        template <>
        inline std::map<ls_initializer, std::string> enum_string<ls_initializer>()
        {
                return
                {
                        { ls_initializer::unit,          "init-unit" },
                        { ls_initializer::quadratic,     "init-quadratic" },
                        { ls_initializer::consistent,    "init-consistent" }
                };
        }

        template <>
        inline std::map<ls_strategy, std::string> enum_string<ls_strategy>()
        {
                return
                {
                        { ls_strategy::backtrack_armijo,         "backtrack-Armijo" },
                        { ls_strategy::backtrack_wolfe,          "backtrack-Wolfe" },
                        { ls_strategy::backtrack_strong_wolfe,   "backtrack-strong-Wolfe" },
                        { ls_strategy::interpolation,            "interp" },
                        { ls_strategy::cg_descent,               "cgdescent" }
                };
        }
}
