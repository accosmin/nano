#pragma once

#include "optim/stoch_optimizer.h"

namespace nano
{
        ///
        /// \brief stochastic gradient (descent) with momentum
        ///
        struct stoch_sgm_t : public stoch_optimizer_t
        {
                NANO_MAKE_CLONABLE(stoch_sgm_t, "stochastic gradient (descent) with momentum")

                ///
                /// \brief constructor
                ///
                stoch_sgm_t(const string_t& configuration = string_t());

                ///
                /// \brief minimize starting from the initial guess x0.
                ///
                virtual state_t minimize(const stoch_params_t&, const problem_t&, const vector_t& x0) const override;

                ///
                /// \brief minimize starting from the initial guess x0 using the given hyper-parameters.
                ///
                state_t minimize(const stoch_params_t&, const problem_t&, const vector_t& x0,
                        const scalar_t alpha0, const scalar_t momentum) const;
        };
}

