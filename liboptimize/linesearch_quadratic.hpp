#pragma once

#include <cmath>

namespace ncv
{
        namespace optimize
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
                        const tscalar x1 = step1.alpha(), f1 = step1.phi(), g1 = step1.gphi();

                        // a x^2 + b x + c
                        const tscalar a = (g0 - (f0 - f1) / (x0 - x1)) / (x0 - x1);
                        const tscalar b = g0 - 2 * a * x0;

                        return -b / (2 * a);
                }
        }
}

