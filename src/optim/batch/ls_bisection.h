#pragma once

#include "ls_step.h"

namespace nano
{
        ///
        /// \brief bisection interpolation in the [step0, step1] line-search interval
        ///
        inline auto ls_bisection(const ls_step_t& step0, const ls_step_t& step1)
        {
                return (step0.alpha() + step1.alpha()) / 2;
        }
}

