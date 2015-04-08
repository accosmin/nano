#pragma once

namespace ncv
{
        namespace optim
        {
                ///
                /// \brief batch optimization methods
                ///
                enum class batch_optimizer
                {
                        GD,                     ///< gradient descent
                        CGD,                    ///< conjugate gradient descent (default version)
                        LBFGS,                  ///< limited-memory BFGS

                        CGD_HS,                 ///< various conjugate gradient descent versions
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
                /// \brief stochastic optimization methods
                ///
                enum class stoch_optimizer
                {
                        SG,                     ///< stochastic gradient
                        SGA,                    ///< stochastic gradient averaging
                        SIA,                    ///< stochastic iterate averaging
                        AG,                     ///< Nesterov's accelerated gradient descent
                        ADAGRAD,                ///< AdaGrad
                        ADADELTA                ///< AdaDelta
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
                        interpolation_bisection,        ///< bisection for zooming
                        interpolation_cubic,            ///< cubic interpolation for zooming

                        // see CG_DESCENT, Hager & Zhang, 2005 - regular and approximate Wolfe only
                        cg_descent                      ///< CG_DESCENT
                };
        }
}

