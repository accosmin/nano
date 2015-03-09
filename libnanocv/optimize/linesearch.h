#pragma once

namespace ncv
{
        namespace optimize
        {
                ///
                /// \brief criterion to choose the line-search step
                ///
                enum class ls_criterion
                {
                        armijo,                 ///< suficient decrease (Armijo)
                        wolfe,                  ///< & suficient curvature (Wolfe)
                        strong_wolfe            ///<
                };

                ///
                /// \brief initial step length strategy
                /// (Nocedal & Wright (numerical optimization 2nd) @ p.59)
                ///
                enum class ls_initializer
                {
                        unit,                   ///< 1.0 (useful for quasi-Newton and Newton methods)
                        consistent,             ///< consistent first-order change in the function
                        quadratic               ///< quadratic local interpolation (previous & current position)
                };

                ///
                /// \brief line-search strategy
                ///
                enum class ls_strategy
                {
                        backtracking,           ///< like implemented in libLBFGS
                        interpolation           ///< (Nocedal & Wright (numerical optimization 2nd) @ p.60-61)
                };
        }
}

