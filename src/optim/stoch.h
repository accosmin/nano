#pragma once

#include "problem.h"
#include "stoch/params.h"

namespace nano
{
        ///
        /// \brief minimize starting from the initial guess x0.
        ///
        NANO_PUBLIC state_t minimize(const stoch_params_t&, const problem_t&, const vector_t& x0);
}
