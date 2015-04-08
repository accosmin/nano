#pragma once

#include "quadratic.hpp"

namespace ncv
{
        namespace optim
        {
                ///
                /// \brief quadratic interpolation in the [step0, step1] line-search interval
                ///     see "Numerical optimization", Nocedal & Wright, 2nd edition, p.58
                ///
                /// NB: using the gradient at step0
                /// NB: the step-length at step0 may be different than zero
                ///
                template
                <
                        typename tstep,
                        typename tscalar = typename tstep::tscalar
                >
                tscalar ls_quadratic(const tstep& step0, const tstep& step1)
                {
                        const tscalar x0 = step0.alpha(), f0 = step0.phi(), g0 = step0.gphi();
                        const tscalar x1 = step1.alpha(), f1 = step1.phi();

                        tscalar a, b, c;
                        optim::quadratic(x0, f0, g0, x1, f1, &a, &b, &c);

                        // OK, return minimum
                        return -b / (2 * a);
                }
        }
}

