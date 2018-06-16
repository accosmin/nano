#pragma once

#include "lsearch_step.h"

namespace nano
{
        ///
        /// \brief compute the initial step length of the line search procedure.
        ///
        class lsearch_init_t
        {
        public:
                virtual ~lsearch_init_t() = default;
                virtual scalar_t get(const solver_state_t&) = 0;
        };

        ///
        /// \brief compute the step length of the line search procedure.
        ///
        class lsearch_step_t
        {
        public:
                virtual ~lsearch_step_t() = default;
                virtual bool get(const function_t&, const scalar_t t0, solver_state_t&) = 0;
        };

        ///
        /// \brief line-search algorithm.
        ///
        class lsearch_t
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
                lsearch_t(const initializer, const strategy, const scalar_t c1, const scalar_t c2);

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
        inline enum_map_t<lsearch_t::initializer> enum_string<lsearch_t::initializer>()
        {
                return
                {
                        { lsearch_t::initializer::unit,                      "unit" },
                        { lsearch_t::initializer::quadratic,                 "quadratic" },
                        { lsearch_t::initializer::consistent,                "consistent" }
                };
        }

        template <>
        inline enum_map_t<lsearch_t::strategy> enum_string<lsearch_t::strategy>()
        {
                return
                {
                        { lsearch_t::strategy::backtrack_armijo,             "back-Armijo" },
                        { lsearch_t::strategy::backtrack_wolfe,              "back-Wolfe" },
                        { lsearch_t::strategy::backtrack_strong_wolfe,       "back-sWolfe" },
                        { lsearch_t::strategy::interpolation,                "interpolation" },
                        { lsearch_t::strategy::cg_descent,                   "cgdescent" }
                };
        }
}
