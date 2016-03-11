#pragma once

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
}

