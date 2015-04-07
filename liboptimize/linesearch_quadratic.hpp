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
                        const tscalar d = step0.alpha() - step1.alpha();
                        const tscalar a = (step0.gphi() - (step0.phi() - step1.phi()) / d) / d;
                        const tscalar b = step0.gphi() - 2 * a * step0.alpha();

                        return -b / (2 * a);
                }
        }
}

