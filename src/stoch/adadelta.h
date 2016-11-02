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

                virtual state_t minimize(const stoch_params_t&, const problem_t&, const vector_t& x0) const override;

                ///
                /// \brief minimize starting from the initial guess x0 using the given hyper-parameters.
                ///
                state_t minimize(const stoch_params_t&, const problem_t&, const vector_t& x0,
                        const scalar_t momentum, const scalar_t epsilon) const;
        };
}

