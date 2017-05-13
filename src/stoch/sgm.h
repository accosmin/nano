#pragma once

#include "stoch_optimizer.h"

namespace nano
{
        ///
        /// \brief stochastic gradient (descent) with momentum
        ///
        struct stoch_sgm_t final : public stoch_optimizer_t
        {
                explicit stoch_sgm_t(const string_t& configuration = string_t());

                state_t minimize(const stoch_params_t&, const function_t&, const vector_t& x0) const override;

                state_t minimize(const stoch_params_t&, const function_t&, const vector_t& x0,
                        const scalar_t alpha0, const scalar_t decay, const scalar_t momentum) const;
        };
}

