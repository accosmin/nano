#pragma once

#include "solver_stoch.h"

namespace nano
{
        ///
        /// \brief averaged stochastic gradient (descent)
        ///     see "Acceleration of stochastic approximation by averaging",
        ///     by Polyak, B. T. and Juditsky, A. B.
        ///
        /// NB: the first-order momentum of the past states is returned instead of the average as in the original paper
        ///     (using the average requires many more iterations).
        ///
        class stoch_asgd_t final : public stoch_solver_t
        {
        public:

                solver_state_t minimize(const stoch_params_t&, const function_t&, const vector_t& x0) const override;

                static solver_state_t minimize(const stoch_params_t&, const function_t&, const vector_t& x0,
                        const scalar_t alpha0, const scalar_t decay, const scalar_t momentum);
        };
}
