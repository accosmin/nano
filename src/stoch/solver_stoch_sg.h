#pragma once

#include "solver_stoch.h"

namespace nano
{
        ///
        /// \brief stochastic gradient (descent)
        ///     see "Minimizing Finite Sums with the Stochastic Average Gradient",
        ///     by Mark Schmidth, Nicolas Le Roux, Francis Bach
        ///
        struct stoch_sg_t final : public stoch_solver_t
        {
                explicit stoch_sg_t(const string_t& params = string_t());

                function_state_t tune(const stoch_params_t&, const function_t&, const vector_t& x0) override;
                function_state_t minimize(const stoch_params_t&, const function_t&, const vector_t& x0) const override;

                static function_state_t minimize(const stoch_params_t&, const function_t&, const vector_t& x0,
                        const scalar_t alpha0, const scalar_t decay);
        };
}
