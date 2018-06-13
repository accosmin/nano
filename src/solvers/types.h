#pragma once

#include "text/cast.h"

namespace nano
{
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
        inline enum_map_t<ls_initializer> enum_string<ls_initializer>()
        {
                return
                {
                        { ls_initializer::unit,          "unit" },
                        { ls_initializer::quadratic,     "quadratic" },
                        { ls_initializer::consistent,    "consistent" }
                };
        }

        template <>
        inline enum_map_t<ls_strategy> enum_string<ls_strategy>()
        {
                return
                {
                        { ls_strategy::backtrack_armijo,         "back-Armijo" },
                        { ls_strategy::backtrack_wolfe,          "back-Wolfe" },
                        { ls_strategy::backtrack_strong_wolfe,   "back-sWolfe" },
                        { ls_strategy::interpolation,            "interpolation" },
                        { ls_strategy::cg_descent,               "cgdescent" }
                };
        }
}
