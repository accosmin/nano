#pragma once

#include "params.h"
#include "optim/problem.h"

namespace nano
{
        ///
        /// \brief gradient descent
        ///
        struct NANO_PUBLIC batch_gd_t
        {
                ///
                /// \brief minimize starting from the initial guess x0
                ///
                state_t operator()(const batch_params_t& param, const problem_t& problem, const vector_t& x0) const;
        };
}

