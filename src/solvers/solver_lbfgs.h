#pragma once

#include "solver.h"

namespace nano
{
        ///
        /// \brief limited memory BGFS (l-BGFS).
        ///     see "Updating Quasi-Newton Matrices with Limited Storage",
        ///     by J. Nocedal, 1980
        ///     see "Numerical Optimization",
        ///     by J. Nocedal, S. Wright, 2006
        ///
        class solver_lbfgs_t final : public solver_t
        {
        public:

                solver_lbfgs_t() = default;

                tuner_t tuner() const final;
                void to_json(json_t&) const final;
                void from_json(const json_t&) final;

                solver_state_t minimize(
                        const size_t max_iterations, const scalar_t epsilon,
                        const function_t&, const vector_t& x0,
                        const logger_t& logger = logger_t()) const final;

        private:

                // attributes
                ls_initializer  m_ls_init{ls_initializer::quadratic};
                ls_strategy     m_ls_strat{ls_strategy::interpolation};
                scalar_t        m_c1{static_cast<scalar_t>(1e-4)};
                scalar_t        m_c2{static_cast<scalar_t>(0.9)};
                size_t          m_history_size{6};                      ///< history size (#previous gradients to consider)
        };
}
