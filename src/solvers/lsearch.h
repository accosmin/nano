#pragma once

#include "lsearch_step.h"

namespace nano
{
        ///
        /// \brief compute the initial step length of the line search procedure.
        ///     see "Numerical optimization", Nocedal & Wright, 2nd edition, p.59
        ///
        class lsearch_init_t
        {
        public:

                ///
                /// \brief destructor
                ///
                virtual ~lsearch_init_t() = default;

                ///
                /// \brief returns the initial step length given the current state
                /// NB: keep track of previous states
                ///
                virtual scalar_t get(const solver_state_t&) = 0;
        };

        ///
        /// \brief compute the step length of the line search procedure.
        ///
        class lsearch_strategy_t
        {
        public:

                ///
                /// \brief constructor
                ///
                lsearch_strategy_t(const scalar_t c1, const scalar_t c2) :
                        m_c1(c1), m_c2(c2)
                {
                }

                ///
                /// \brief destructor
                ///
                virtual ~lsearch_strategy_t() = default;

                ///
                /// \brief returns the step length given the current state
                ///
                virtual lsearch_step_t get(const lsearch_step_t& step0, const scalar_t t0) = 0;

                ///
                /// \brief access functions
                ///
                auto c1() const { return m_c1; }
                auto c2() const { return m_c2; }

        protected:

                // attributes
                scalar_t                m_c1;           ///< sufficient decrease rate
                scalar_t                m_c2;           ///< sufficient curvature
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
                        linear,                         ///< consistent first-order change in the function
                        quadratic,                      ///< quadratic local interpolation (previous & current position)
                };

                ///
                /// \brief line-search strategy
                ///
                enum class strategy
                {
                        backtrack_armijo,               ///< backtracking with sufficient decrease (Armijo)
                        backtrack_wolfe,                ///< + backtracking with suficient curvature (Wolfe)
                        backtrack_swolfe,               ///< + backtracking with sufficient curvature (strong Wolfe)
                        cg_descent,                     ///< CG_DESCENT (regular and approximation Wolfe)
                        more_thuente,                   ///< More & Thunte (strong Wolfe)
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

                // attributes
                std::unique_ptr<lsearch_init_t>         m_initializer;
                std::unique_ptr<lsearch_strategy_t>     m_strategy;
        };

        template <>
        inline enum_map_t<lsearch_t::initializer> enum_string<lsearch_t::initializer>()
        {
                return
                {
                        { lsearch_t::initializer::unit,                 "unit" },
                        { lsearch_t::initializer::linear,               "linear" },
                        { lsearch_t::initializer::quadratic,            "quadratic" }
                };
        }

        template <>
        inline enum_map_t<lsearch_t::strategy> enum_string<lsearch_t::strategy>()
        {
                return
                {
                        { lsearch_t::strategy::backtrack_armijo,        "back-armijo" },
                        { lsearch_t::strategy::backtrack_wolfe,         "back-wolfe" },
                        { lsearch_t::strategy::backtrack_swolfe,        "back-swolfe" },
                        { lsearch_t::strategy::cg_descent,              "cg-descent" },
                        { lsearch_t::strategy::more_thuente,            "more-thuente" }
                };
        }
}
