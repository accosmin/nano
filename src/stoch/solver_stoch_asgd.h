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
        struct stoch_asgd_t final : public stoch_solver_t
        {
                explicit stoch_asgd_t(const string_t& configuration = string_t());

                function_state_t tune(const stoch_params_t&, const function_t&, const vector_t& x0) override;
                function_state_t minimize(const stoch_params_t&, const function_t&, const vector_t& x0) const override;

                static function_state_t minimize(const stoch_params_t&, const function_t&, const vector_t& x0,
                        const scalar_t alpha0, const scalar_t decay, const scalar_t momentum);
        };
}
