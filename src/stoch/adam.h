#pragma once

#include "stoch_optimizer.h"

namespace nano
{
        ///
        /// \brief stochastic Adam,
        ///     see "Adam: A method for stochastic optimization", by Diederik P. Kingma & Jimmy Lei Ba
        ///
        struct stoch_adam_t final : public stoch_optimizer_t
        {
                explicit stoch_adam_t(const string_t& configuration = string_t());

                virtual rstoch_optimizer_t clone() const override;

                virtual state_t minimize(const stoch_params_t&, const problem_t&, const vector_t& x0) const override;

                ///
                /// \brief minimize starting from the initial guess x0 using the given hyper-parameters.
                ///
                state_t minimize(const stoch_params_t&, const problem_t&, const vector_t& x0,
                        const scalar_t alpha0, const scalar_t epsilon) const;
        };
}

