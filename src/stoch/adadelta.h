#pragma once

#include "solver_stoch.h"

namespace nano
{
        ///
        /// \brief stochastic AdaDelta,
        ///     see "ADADELTA: An Adaptive Learning Rate Method", by Matthew D. Zeiler
        ///
        struct stoch_adadelta_t final : public stoch_solver_t
        {
                explicit stoch_adadelta_t(const string_t& configuration = string_t());

                function_state_t minimize(const stoch_params_t&, const function_t&, const vector_t& x0) const override;

                function_state_t minimize(const stoch_params_t&, const function_t&, const vector_t& x0,
                        const scalar_t momentum, const scalar_t epsilon) const;
        };
}
