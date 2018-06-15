#pragma once

#include "linesearch_step.h"

namespace nano
{
        ///
        /// \brief line-search algorithm.
        ///
        class lineasearch_t
        {
        public:

                ///
                /// \brief initial step length strategy
                ///     see "Numerical optimization", Nocedal & Wright, 2nd edition, p.59
                ///
                enum class initializer
                {
                        unit,                           ///< 1.0 (useful for quasi-Newton and Newton methods)
                        consistent,                     ///< consistent first-order change in the function
                        quadratic                       ///< quadratic local interpolation (previous & current position)
                };

                ///
                /// \brief line-search strategy
                ///
                enum class strategy
                {
                        backtrack_armijo,               ///< backtracking with sufficient decrease (Armijo)
                        backtrack_wolfe,                ///< + backtracking with suficient curvature (Wolfe)
                        backtrack_strong_wolfe,         ///< + backtracking with sufficient curvature (strong Wolfe)

                        // see "Numerical optimization", Nocedal & Wright, 2nd edition, p.60-61 - strong Wolfe only
                        interpolation,                  ///< bisection/quadratic/cubic for zooming

                        // see CG_DESCENT, Hager & Zhang, 2005 - regular and approximate Wolfe only
                        cg_descent                      ///< CG_DESCENT
                };

                ///
                /// \brief constructor
                ///
                linesearch_t(const initializer, const strategy, const scalar_t c1, const scalar_t c2);

                ///
                /// \brief update the current state
                ///
                bool operator()(const function_t& function, solver_state_t& state);

        private:

                ///
                /// \brief compute the initial step length
                ///
                scalar_t t0(const solver_state_t& cstate);

        private:

                // attributes
                initializer     m_initializer;
                strategy        m_strategy;
                bool            m_first{true};  ///< check if first iteration
                scalar_t        m_prevf;        ///< previous function evaluation
                scalar_t        m_prevt0;       ///< previous step length
                scalar_t        m_prevdg;       ///< previous direction dot product
                scalar_t        m_c1;           ///< sufficient decrease rate
                scalar_t        m_c2;           ///< sufficient curvature
        };

        template <>
        inline enum_map_t<linesearch_t::initializer> enum_string<linesearch_t::initializer>()
        {
                return
                {
                        { linesearch_t::initializer::unit,                      "unit" },
                        { linesearch_t::initializer::quadratic,                 "quadratic" },
                        { linesearch_t::initializer::consistent,                "consistent" }
                };
        }

        template <>
        inline enum_map_t<linesearch_t::strategy> enum_string<linesearch_t::strategy>()
        {
                return
                {
                        { linesearch_t::strategy::backtrack_armijo,             "back-Armijo" },
                        { linesearch_t::strategy::backtrack_wolfe,              "back-Wolfe" },
                        { linesearch_t::strategy::backtrack_strong_wolfe,       "back-sWolfe" },
                        { linesearch_t::strategy::interpolation,                "interpolation" },
                        { linesearch_t::strategy::cg_descent,                   "cgdescent" }
                };
        }
}
