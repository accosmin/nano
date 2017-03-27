#pragma once

#include "stoch_optimizer.h"

namespace nano
{
        ///
        /// \brief stochastic AdaDelta,
        ///     see "ADADELTA: An Adaptive Learning Rate Method", by Matthew D. Zeiler
        ///
        struct stoch_adadelta_t final : public stoch_optimizer_t
        {
                explicit stoch_adadelta_t(const string_t& configuration = string_t());

                state_t minimize(const stoch_params_t&, const function_t&, const vector_t& x0) const override;

                state_t minimize(const stoch_params_t&, const function_t&, const vector_t& x0,
                        const scalar_t alpha0, const scalar_t decay,
                        const scalar_t momentum, const scalar_t epsilon) const;
        };
}
