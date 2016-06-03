#pragma once

#include "ls_step.hpp"
#include "math/quadratic.hpp"

namespace nano
{
        ///
        /// \brief quadratic interpolation in the [step0, step1] line-search interval
        ///     see "Numerical optimization", Nocedal & Wright, 2nd edition, p.58
        ///
        /// NB: using the gradient at step0
        /// NB: the step-length at step0 may be different than zero
        ///
        inline auto ls_quadratic(const ls_step_t& step0, const ls_step_t& step1)
        {
                const scalar_t x0 = step0.alpha(), f0 = step0.phi(), g0 = step0.gphi();
                const scalar_t x1 = step1.alpha(), f1 = step1.phi();

                const quadratic_t<scalar_t> q(x0, f0, g0, x1, f1);

                scalar_t min = std::numeric_limits<scalar_t>::infinity();

                if (q)
                {
                        q.extremum(min);
                }

                return min;
        }
}

