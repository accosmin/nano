#pragma once

#include "solver.h"
#include "lsearch.h"

namespace nano
{
        ///
        /// \brief quasi-newton methods.
        ///     see "Numerical Optimization",
        ///     by J. Nocedal, S. Wright, 2006
        ///     see "Introductory Lectures on Convex Optimization (Applied Optimization)",
        ///     by Y. Nesterov, 2013
        ///
        class solver_quasi_t final : public solver_t
        {
        public:

                solver_quasi_t() = default;

                tuner_t tuner() const final;
                void to_json(json_t&) const final;
                void from_json(const json_t&) final;

                solver_state_t minimize(
                        const size_t max_iterations, const scalar_t epsilon,
                        const function_t&, const vector_t& x0,
                        const logger_t& logger = logger_t()) const final;

        private:

                // attributes
                lsearch_t::initializer  m_init{lsearch_t::initializer::quadratic};
                lsearch_t::strategy     m_strat{lsearch_t::strategy::interpolate};
                scalar_t                m_c1{static_cast<scalar_t>(1e-4)};
                scalar_t                m_c2{static_cast<scalar_t>(0.9)};
        };
}
