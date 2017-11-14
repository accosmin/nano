#pragma once

#include "solver_stoch.h"

namespace nano
{
        ///
        /// \brief stochastic AdaDelta,
        ///     see "ADADELTA: An Adaptive Learning Rate Method", by Matthew D. Zeiler
        ///
        class stoch_adadelta_t final : public stoch_solver_t
        {
        public:

                solver_state_t minimize(const stoch_params_t&, const function_t&, const vector_t& x0) const override;

                static solver_state_t minimize(const stoch_params_t&, const function_t&, const vector_t& x0,
                        const scalar_t momentum, const scalar_t epsilon);
        };
}
