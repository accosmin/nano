#pragma once

#include "lsearch.h"

namespace nano
{
        ///
        /// \brief backtracking line-search that stops when the Wolfe condition is satisfied,
        ///
        /// \brief backtracking line-search that stops when the Armijo condition is satisfied,
        ///     see "Numerical optimization", Nocedal & Wright, 2nd edition
        ///     see libLBFGS
        ///
        class lsearch_backtrack_armijo_t final : public lsearch_strategy_t
        {
        public:

                lsearch_backtrack_armijo_t() = default;
                bool get(const solver_state_t& state0, const scalar_t t0, solver_state_t& state) final;

        private:

                // attributes
                scalar_t        m_decrement{static_cast<scalar_t>(0.5)};///<
        };

        ///
        /// \brief backtracking line-search that stops when the Wolfe condition is satisfied,
        ///     see "Numerical optimization", Nocedal & Wright, 2nd edition
        ///     see libLBFGS
        ///
        class lsearch_backtrack_wolfe_t final : public lsearch_strategy_t
        {
        public:

                lsearch_backtrack_wolfe_t() = default;
                bool get(const solver_state_t& state0, const scalar_t t0, solver_state_t& state) final;

        private:

                // attributes
                scalar_t        m_decrement{static_cast<scalar_t>(0.5)};///<
                scalar_t        m_increment{static_cast<scalar_t>(2.1)};///<
        };

        ///
        /// \brief backtracking line-search that stops when the Wolfe condition is satisfied,
        ///     see "Numerical optimization", Nocedal & Wright, 2nd edition
        ///     see libLBFGS
        ///
        class lsearch_backtrack_swolfe_t final : public lsearch_strategy_t
        {
        public:

                lsearch_backtrack_swolfe_t() = default;
                bool get(const solver_state_t& state0, const scalar_t t0, solver_state_t& state) final;

        private:

                // attributes
                scalar_t        m_decrement{static_cast<scalar_t>(0.5)};///<
                scalar_t        m_increment{static_cast<scalar_t>(2.1)};///<
        };
}
