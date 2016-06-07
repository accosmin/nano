#pragma once

#include "params.h"
#include "optim/problem.h"

namespace nano
{
        ///
        /// \brief stochastic gradient (descent) with momentum
        ///
        struct NANO_PUBLIC stoch_sgm_t
        {
                ///
                /// \brief minimize starting from the initial guess x0
                ///
                state_t operator()(const stoch_params_t& param, const problem_t& problem, const vector_t& x0) const;

                ///
                /// \brief minimize starting from the initial guess x0
                ///
                state_t operator()(const stoch_params_t& param, const problem_t& problem, const vector_t& x0,
                        const scalar_t alpha0, const scalar_t decay, const scalar_t momentum) const;
        };
}

