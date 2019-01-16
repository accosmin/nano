#pragma once

#include "state.h"

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
                /// NB: may keep track of the previous states
                ///
                scalar_t get(const solver_state_t& state)
                {
                        return get(state, m_iteration ++);
                }

        private:

                virtual scalar_t get(const solver_state_t&, const int iteration) = 0;

                // attributes
                int             m_iteration{0}; ///<
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
                lsearch_strategy_t() = default;

                ///
                /// \brief destructor
                ///
                virtual ~lsearch_strategy_t() = default;

                ///
                /// \brief
                ///
                virtual bool get(const solver_state_t& state0, const scalar_t t0, solver_state_t&) = 0;

                ///
                /// \brief minimum allowed line-search step
                ///
                static scalar_t stpmin()
                {
                        return scalar_t(10) * std::numeric_limits<scalar_t>::epsilon();
                }

                ///
                /// \brief maximum allowed line-search step
                ///
                static scalar_t stpmax()
                {
                        return scalar_t(1) / stpmin();
                }

                ///
                /// \brief change parameters
                ///
                auto& c1(const scalar_t c1) { m_c1 = c1; return *this; }
                auto& c2(const scalar_t c2) { m_c2 = c2; return *this; }
                auto& max_iterations(const int max_iterations) { m_max_iterations = max_iterations; return *this; }

                ///
                /// \brief access functions
                ///
                auto c1() const { return m_c1; }
                auto c2() const { return m_c2; }
                auto max_iterations() const { return m_max_iterations; }

        private:

                // attributes
                scalar_t        m_c1{static_cast<scalar_t>(1e-4)};      ///< sufficient decrease rate
                scalar_t        m_c2{static_cast<scalar_t>(0.1)};       ///< sufficient curvature
                int             m_max_iterations{40};                   ///< #maximum iterations
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
                ///     see CG_DESCENT
                ///
                enum class initializer
                {
                        unit,                           ///< 1.0 (useful for quasi-Newton and Newton methods)
                        linear,                         ///< consistent first-order change in the function
                        quadratic,                      ///< quadratic local interpolation (previous & current position)
                        cg_descent,                     ///< CG_DESCENT
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
                bool operator()(solver_state_t& state);

        private:

                // attributes
                std::unique_ptr<lsearch_init_t>         m_initializer;  ///<
                std::unique_ptr<lsearch_strategy_t>     m_strategy;     ///<
        };

        template <>
        inline enum_map_t<lsearch_t::initializer> enum_string<lsearch_t::initializer>()
        {
                return
                {
                        { lsearch_t::initializer::unit,                 "unit" },
                        { lsearch_t::initializer::linear,               "linear" },
                        { lsearch_t::initializer::quadratic,            "quadratic" },
                        { lsearch_t::initializer::cg_descent,           "cg-descent" }
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
