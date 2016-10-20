#pragma once

#include "stoch_optimizer.h"

namespace nano
{
        ///
        /// \brief stochastic gradient (descent)
        ///     see "Minimizing Finite Sums with the Stochastic Average Gradient",
        ///     by Mark Schmidth, Nicolas Le Roux, Francis Bach
        ///
        struct stoch_sg_t final : public stoch_optimizer_t
        {
                explicit stoch_sg_t(const string_t& configuration = string_t());

                virtual rstoch_optimizer_t clone(const string_t& configuration) const override;
                virtual rstoch_optimizer_t clone() const override;

                virtual state_t minimize(const stoch_params_t&, const problem_t&, const vector_t& x0) const override;

                ///
                /// \brief minimize starting from the initial guess x0 using the given hyper-parameters.
                ///
                state_t minimize(const stoch_params_t&, const problem_t&, const vector_t& x0,
                        const scalar_t alpha0, const scalar_t decay) const;
        };
}

