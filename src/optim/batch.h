#pragma once

#include "problem.h"
#include "batch/params.h"

namespace nano
{
        ///
        /// \brief minimize starting from the initial guess x0.
        ///
        NANO_PUBLIC state_t minimize(const batch_params_t&, const problem_t&, const vector_t& x0);
}

