#pragma once

#include "stoch_optimizer.h"

namespace nano
{
        ///
        /// \brief stochastic AdaGrad
        ///     see "Adaptive subgradient methods for online learning and stochastic optimization"
        ///     by J. C. Duchi, E. Hazan, and Y. Singer
        ///
        ///     see http://xcorr.net/2014/01/23/adagrad-eliminating-learning-rates-in-stochastic-gradient-descent/
        ///
        struct stoch_adagrad_t final : public stoch_optimizer_t
        {
                explicit stoch_adagrad_t(const string_t& configuration = string_t());

                state_t minimize(const stoch_params_t&, const function_t&, const vector_t& x0) const override;

                state_t minimize(const stoch_params_t&, const function_t&, const vector_t& x0,
                        const scalar_t alpha0, const scalar_t epsilon) const;
        };
}
