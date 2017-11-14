#pragma once

#include "solver_stoch.h"

namespace nano
{
        ///
        /// \brief stochastic Adam,
        ///     see "Adam: A method for stochastic optimization", by Diederik P. Kingma & Jimmy Lei Ba
        ///
        class stoch_adam_t final : public stoch_solver_t
        {
        public:

                function_state_t minimize(const stoch_params_t&, const function_t&, const vector_t& x0) const override;

                static function_state_t minimize(const stoch_params_t&, const function_t&, const vector_t& x0,
                        const scalar_t alpha0, const scalar_t decay, const scalar_t epsilon,
                        const scalar_t beta1, const scalar_t beta2);
        };
}
