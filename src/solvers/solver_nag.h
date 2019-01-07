#pragma once

#include "solver.h"
#include "lsearch.h"

namespace nano
{
        ///
        /// \brief Nesterov's accelerated gradient descent (with variations)
        ///     see "Adaptive Restart for Accelerated Gradient Schemes",
        ///     by Brendan O’Donoghue & Emmanuel Candes, 2013
        ///     see "Introductory lectures on convex optimization: A basic course",
        ///     by Y. Nesterov, 2004
        ///
        /// NB: the implementation forces the chosen direction to be a descent direction
        ///     (aka the gradient restart scheme)
        ///
        class solver_nag_t final : public solver_t
        {
        public:

                solver_nag_t() = default;

                tuner_t tuner() const final;
                void to_json(json_t&) const final;
                void from_json(const json_t&) final;

                solver_state_t minimize(
                        const size_t max_iterations, const scalar_t epsilon,
                        const solver_function_t&, const vector_t& x0, const logger_t&) const final;

        private:

                // attributes
                lsearch_t::initializer  m_init{lsearch_t::initializer::unit};
                lsearch_t::strategy     m_strat{lsearch_t::strategy::more_thuente};
                scalar_t                m_c1{static_cast<scalar_t>(1e-4)};
                scalar_t                m_c2{static_cast<scalar_t>(9e-1)};
                scalar_t                m_q{static_cast<scalar_t>(0.0)};
        };
}
