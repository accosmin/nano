#pragma once

#include "ls_step.h"
#include "math/cubic.h"
#include <utility>

namespace nano
{
        ///
        /// \brief cubic interpolation in the [step0, step1] line-search interval
        ///     see "Numerical optimization", Nocedal & Wright, 2nd edition, p.59
        ///
        inline auto ls_cubic(const ls_step_t& step0, const ls_step_t& step1)
        {
                const scalar_t x0 = step0.alpha(), f0 = step0.phi(), g0 = step0.gphi();
                const scalar_t x1 = step1.alpha(), f1 = step1.phi(), g1 = step1.gphi();

                const cubic_t<scalar_t> c(x0, f0, g0, x1, f1, g1);

                scalar_t min1 = std::numeric_limits<scalar_t>::infinity();
                scalar_t min2 = std::numeric_limits<scalar_t>::infinity();

                if (c)
                {
                        c.extremum(min1, min2);
                }

                return std::make_pair(min1, min2);
        }
}

